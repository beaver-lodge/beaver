defmodule Beaver.Execution do
  @moduledoc false
  alias Beaver.{MLIR, Native}
  alias Beaver.MLIR.CAPI
  alias Beaver.Execution.Profile

  def run(ctx, command, arguments, opts \\ []) do
    profile = opts[:profile]
    strategy = Keyword.get(opts, :strategy, :worker)

    Profile.measure(profile, phase(command), fn ->
      {{outcome, costs}, actual_strategy} =
        case strategy do
          :worker ->
            id = CAPI.beaver_raw_execution_worker(ctx.ref, command, arguments, not is_nil(profile))
            {await(id, opts), :worker}

          :dirty ->
            {CAPI.beaver_raw_execution_dirty(ctx.ref, command, arguments, not is_nil(profile)), :dirty}

          :auto ->
            case CAPI.beaver_raw_execution_auto(ctx.ref, command, arguments, not is_nil(profile)) do
              id when is_reference(id) -> {await(id, opts), :worker}
              result -> {result, :dirty}
            end

          other ->
            raise ArgumentError, "unsupported execution strategy: #{inspect(other)}"
        end

      Profile.native(profile, command, actual_strategy, costs)

      case outcome do
        {:ok, result} -> Native.check!(result)
        {:error, reason} -> raise ArgumentError, "native #{command} failed: #{reason}"
      end
    end)
  end

  # Correlate nested completions. Wait indefinitely unless the interface query
  # explicitly requests a timeout; timing out does not cancel native work.
  defp await(id, opts) do
    receive do
      {^id, result} -> result
    after
      Keyword.get(opts, :timeout, :infinity) -> raise "timed out waiting for native execution"
    end
  end

  defp phase(:pass), do: :native_pass
  defp phase(:snapshot), do: :batch_read
  defp phase(:update), do: :batch_update
  defp phase(command), do: command

  @doc false
  def executor_stats(%MLIR.Context{ref: ref}) do
    keys = [:capacity, :threads, :running, :waiting, :queued, :peak_running, :peak_threads]
    Map.new(Enum.zip(keys, CAPI.beaver_raw_executor_snapshot(ref) |> Tuple.to_list()))
  end
end
