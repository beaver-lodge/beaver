defmodule Beaver.Execution.Profile do
  @moduledoc """
  Bounded, opt-in execution costs shared by passes, verification, rewrites and
  IR batches. Pass `profile: profile` inside `capture/1` to attribute nested work.

  Phase durations are inclusive. Native callback wait is summed across threads
  and may exceed wall time; it must not be subtracted from wall time as CPU time.
  Uninstrumented execution does not allocate a receipt or read profiling clocks.
  Queue time covers Beaver's executor; dirty-scheduler queue time is not sampled.
  Completion delivery includes result packaging and the BEAM caller's wakeup.
  CPU time and peak RSS describe the whole VM process, including concurrent work.
  """
  defstruct [:table]
  @type t :: %__MODULE__{table: :ets.tid()}

  def capture(fun) when is_function(fun, 1) do
    profile = %__MODULE__{table: :ets.new(__MODULE__, [:public, :set, write_concurrency: true])}
    cpu = Beaver.MLIR.CAPI.beaver_raw_process_cpu_time()
    rss = Beaver.MLIR.CAPI.beaver_raw_peak_rss()
    started = Beaver.MLIR.CAPI.beaver_raw_execution_clock()
    reductions = reductions()

    try do
      result = fun.(profile)

      receipt = %{
        schema_version: 1,
        duration_ns: Beaver.MLIR.CAPI.beaver_raw_execution_clock() - started,
        caller_reductions: reductions() - reductions,
        vm_cpu_time_ns: max(Beaver.MLIR.CAPI.beaver_raw_process_cpu_time() - cpu, 0),
        vm_peak_rss_before_bytes: rss,
        vm_peak_rss_after_bytes: Beaver.MLIR.CAPI.beaver_raw_peak_rss(),
        phases: entries(profile, :phase),
        native: entries(profile, :native)
      }

      {result, receipt}
    after
      :ets.delete(profile.table)
    end
  end

  def measure(nil, _phase, fun), do: fun.()

  def measure(%__MODULE__{} = profile, phase, fun) do
    started = Beaver.MLIR.CAPI.beaver_raw_execution_clock()

    try do
      fun.()
    after
      record(profile, phase, Beaver.MLIR.CAPI.beaver_raw_execution_clock() - started)
    end
  end

  @doc false
  def record(profile, phase, duration), do: add(profile, {:phase, phase}, [1, duration])

  def native(nil, _command, _strategy, _costs), do: :ok

  def native(profile, command, strategy, {queue, duration, count, wait, setup, completed}) do
    delivery = max(Beaver.MLIR.CAPI.beaver_raw_execution_clock() - completed, 0)
    add(profile, {:native, {command, strategy}}, [1, queue, duration, count, wait, setup, delivery])
  end

  defp add(profile, key, values) do
    :ets.update_counter(
      profile.table,
      key,
      Enum.with_index(values, 2) |> Enum.map(fn {v, i} -> {i, v} end),
      List.to_tuple([key | List.duplicate(0, length(values))])
    )

    :ok
  end

  defp entries(profile, kind) do
    for row <- :ets.tab2list(profile.table), {^kind, key} <- [elem(row, 0)], into: %{} do
      names =
        if kind == :phase,
          do: [:count, :duration_ns],
          else: ~w(count queue_wait_ns duration_ns callback_count callback_wait_ns setup_ns completion_delivery_ns)a

      key = if kind == :native, do: key |> Tuple.to_list() |> Enum.join("/"), else: key
      {key, Map.new(Enum.zip(names, tl(Tuple.to_list(row))))}
    end
  end

  defp reductions, do: self() |> Process.info(:reductions) |> elem(1)
end
