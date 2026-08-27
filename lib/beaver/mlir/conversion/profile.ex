defmodule Beaver.MLIR.Conversion.Profile do
  @moduledoc """
  Builds a bounded receipt for one dialect conversion.

  Profiling is explicit: ordinary conversion does not allocate callback
  aggregates or walk the IR inventory. The receipt retains one aggregate per
  callback kind, never one record per callback invocation.
  """

  alias Beaver.MLIR

  @schema_version 1
  @memory_sample_interval 256

  @type receipt() :: map()
  @type state() :: map()

  @doc false
  @spec start(MLIR.Conversion.mode(), MLIR.Conversion.conversion_ir()) :: state()
  def start(mode, ir) do
    %{
      mode: mode,
      started_at: monotonic_ns(),
      process_cpu_time_ns: MLIR.CAPI.beaver_raw_process_cpu_time(),
      peak_rss_bytes: MLIR.CAPI.beaver_raw_peak_rss(),
      reductions: reductions(),
      process_memory_bytes: process_memory_bytes(),
      peak_process_memory_bytes: process_memory_bytes(),
      ir_before: inventory(ir),
      callbacks: %{},
      callback_count: 0
    }
  end

  @doc false
  @spec record_callback(state(), atom(), non_neg_integer(), non_neg_integer()) :: state()
  def record_callback(state, kind, service_ns, native_wait_ns)
      when is_atom(kind) and service_ns >= 0 and native_wait_ns >= service_ns do
    callback_count = state.callback_count + 1

    peak_process_memory_bytes =
      if rem(callback_count, @memory_sample_interval) == 0 do
        max(state.peak_process_memory_bytes, process_memory_bytes())
      else
        state.peak_process_memory_bytes
      end

    summary = %{
      count: 1,
      service_ns: service_ns,
      max_service_ns: service_ns,
      native_wait_ns: native_wait_ns,
      max_native_wait_ns: native_wait_ns
    }

    %{
      state
      | callbacks:
          Map.update(state.callbacks, kind, summary, fn current ->
            %{
              count: current.count + 1,
              service_ns: current.service_ns + service_ns,
              max_service_ns: max(current.max_service_ns, service_ns),
              native_wait_ns: current.native_wait_ns + native_wait_ns,
              max_native_wait_ns: max(current.max_native_wait_ns, native_wait_ns)
            }
          end),
        peak_process_memory_bytes: peak_process_memory_bytes,
        callback_count: callback_count
    }
  end

  @doc false
  @spec finish(state(), MLIR.Conversion.result(), MLIR.Conversion.conversion_ir(), term()) ::
          receipt()
  def finish(state, result, ir, native_profile) do
    memory_after = process_memory_bytes()
    process_cpu_time_after_ns = MLIR.CAPI.beaver_raw_process_cpu_time()
    peak_rss_after_bytes = MLIR.CAPI.beaver_raw_peak_rss()
    native = decode_native_profile(native_profile)
    callbacks = callback_summaries(state.callbacks)
    native_wait_ns = Enum.sum(Enum.map(callbacks, & &1["native_wait_sum_ns"]))
    max_native_wait_ns = Enum.max(Enum.map(callbacks, & &1["max_native_wait_ns"]), fn -> 0 end)
    beam_service_ns = Enum.sum(Enum.map(callbacks, & &1["beam_service_ns"]))
    boundary_overhead_sum_ns = max(native_wait_ns - beam_service_ns, 0)

    unattributed_residual_ns =
      max(native.duration_ns - native.target_lock_wait_ns - native_wait_ns, 0)

    hotspots =
      hotspots(
        callbacks,
        unattributed_residual_ns,
        native.target_lock_wait_ns
      )

    %{
      "schema_version" => @schema_version,
      "mode" => Atom.to_string(state.mode),
      "status" => status(result),
      "duration_ns" => elapsed_ns(state.started_at),
      "process_cpu_time_ns" => max(process_cpu_time_after_ns - state.process_cpu_time_ns, 0),
      "rss" => %{
        "peak_before_bytes" => state.peak_rss_bytes,
        "peak_after_bytes" => peak_rss_after_bytes,
        "peak_delta_bytes" => max(peak_rss_after_bytes - state.peak_rss_bytes, 0)
      },
      "beam" => %{
        "reductions" => max(reductions() - state.reductions, 0),
        "process_memory_before_bytes" => state.process_memory_bytes,
        "process_memory_after_bytes" => memory_after,
        "peak_process_memory_bytes" => max(state.peak_process_memory_bytes, memory_after),
        "memory_sample_interval_callbacks" => @memory_sample_interval,
        "callback_service_ns" => beam_service_ns,
        "callback_count" => state.callback_count,
        "max_in_flight" => if(state.callback_count == 0, do: 0, else: 1)
      },
      "native" => %{
        "duration_ns" => native.duration_ns,
        "target_lock_wait_ns" => native.target_lock_wait_ns,
        "callback_wait_sum_ns" => native_wait_ns,
        "callback_wait_max_ns" => max_native_wait_ns,
        "unattributed_residual_ns" => unattributed_residual_ns
      },
      "boundary_overhead_sum_ns" => boundary_overhead_sum_ns,
      "hotspots" => hotspots,
      "callbacks" => callbacks,
      "ir" => %{
        "before" => state.ir_before,
        "after" => inventory(ir)
      }
    }
  end

  defp decode_native_profile({duration_ns, target_lock_wait_ns})
       when is_integer(duration_ns) and is_integer(target_lock_wait_ns) do
    %{
      duration_ns: duration_ns,
      target_lock_wait_ns: target_lock_wait_ns
    }
  end

  defp callback_summaries(callbacks) do
    callbacks
    |> Map.keys()
    |> Enum.sort()
    |> Enum.map(fn kind ->
      callback = Map.fetch!(callbacks, kind)

      %{
        "kind" => Atom.to_string(kind),
        "count" => callback.count,
        "beam_service_ns" => callback.service_ns,
        "max_beam_service_ns" => callback.max_service_ns,
        "native_wait_sum_ns" => callback.native_wait_ns,
        "max_native_wait_ns" => callback.max_native_wait_ns,
        "boundary_overhead_sum_ns" => max(callback.native_wait_ns - callback.service_ns, 0)
      }
    end)
  end

  defp hotspots(callbacks, native_compute_ns, target_lock_wait_ns) do
    callback_sources =
      Enum.flat_map(callbacks, fn callback ->
        kind = callback["kind"]

        [
          hotspot("beam_callback_service", kind, callback["beam_service_ns"]),
          hotspot("callback_boundary_wait", kind, callback["boundary_overhead_sum_ns"])
        ]
      end)

    [
      hotspot("native_unattributed_residual", nil, native_compute_ns),
      hotspot("target_lock_wait", nil, target_lock_wait_ns)
      | callback_sources
    ]
    |> Enum.reject(&(&1["duration_ns"] == 0))
    |> Enum.sort_by(&{-&1["duration_ns"], &1["source"], &1["callback_kind"] || ""})
    |> Enum.take(3)
  end

  defp hotspot(source, callback_kind, duration_ns) do
    %{
      "source" => source,
      "callback_kind" => callback_kind,
      "duration_ns" => duration_ns
    }
  end

  defp inventory(ir) do
    operation = MLIR.Operation.from_module(ir)
    {operations, modules, functions} = MLIR.CAPI.beaver_raw_operation_inventory(operation.ref)

    %{
      "operations" => operations,
      "modules" => modules,
      "functions" => functions
    }
  end

  defp status({:ok, _ir, _diagnostics}), do: "ok"
  defp status({:error, _error}), do: "error"

  defp reductions do
    {:reductions, reductions} = Process.info(self(), :reductions)
    reductions
  end

  defp process_memory_bytes do
    {:memory, memory} = Process.info(self(), :memory)
    memory
  end

  defp monotonic_ns, do: System.monotonic_time(:nanosecond)
  defp elapsed_ns(started_at), do: max(monotonic_ns() - started_at, 0)
end
