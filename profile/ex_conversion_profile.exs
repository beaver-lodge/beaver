defmodule Beaver.Profile.ExConversion do
  @moduledoc false

  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Ex
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Dialect.Ex, as: ExDialect

  def run(argv) do
    argv = if List.first(argv) == "--", do: tl(argv), else: argv

    {opts, [], []} =
      OptionParser.parse(argv,
        strict: [
          functions: :integer,
          iterations: :integer,
          warmup: :integer,
          max_overhead_percent: :float
        ]
      )

    functions = positive!(opts[:functions] || 64, "functions")
    iterations = positive!(opts[:iterations] || 7, "iterations")
    warmup = non_negative!(opts[:warmup] || 2, "warmup")
    max_overhead_percent = optional_positive!(opts[:max_overhead_percent], "max-overhead-percent")
    assembly = assembly(functions)

    if warmup > 0 do
      Enum.each(1..warmup, fn _ -> measure(assembly, false) end)
    end

    measurements =
      Enum.flat_map(1..iterations, fn _ ->
        [measure(assembly, false), measure(assembly, true)]
      end)

    baseline = Enum.filter(measurements, &(&1.mode == "baseline"))
    profiled = Enum.filter(measurements, &(&1.mode == "profiled"))
    fingerprints = measurements |> Enum.map(& &1.fingerprint) |> Enum.uniq()

    if length(fingerprints) != 1 do
      raise "profiled and baseline conversions produced different IR fingerprints"
    end

    overhead_ratio = paired_overhead_ratio(measurements)

    report = %{
      "schema_version" => 1,
      "fixture" => %{
        "kind" => "synthetic_ex_conversion",
        "functions" => functions,
        "input_operations" => 1 + functions * 4
      },
      "iterations" => iterations,
      "warmup" => warmup,
      "fingerprint" => hd(fingerprints),
      "baseline" => summary(baseline),
      "profiled" => summary(profiled),
      "observer_overhead_ratio" => overhead_ratio,
      "observer_overhead_percent" => overhead_ratio * 100.0,
      "max_overhead_percent" => max_overhead_percent,
      "receipt" => List.last(profiled).receipt
    }

    report |> JSON.encode!() |> IO.puts()

    if max_overhead_percent && report["observer_overhead_percent"] > max_overhead_percent do
      raise "observer overhead #{report["observer_overhead_percent"]}% exceeds #{max_overhead_percent}%"
    end
  end

  defp measure(assembly, profile?) do
    ctx = MLIR.Context.create()

    try do
      assert_success!(Beaver.Slang.load(ctx, ExDialect), "loading the Ex dialect")
      module = assembly |> MLIR.Module.create!(ctx: ctx) |> MLIR.verify!()
      rss_before = rss_bytes()
      cpu_before_ns = MLIR.CAPI.beaver_raw_process_cpu_time()
      started_at = System.monotonic_time(:nanosecond)

      {converted, receipt} =
        if profile? do
          Plan.profile!(Ex.plan(), module)
        else
          {Plan.run!(Ex.plan(), module), nil}
        end

      duration_ns = max(System.monotonic_time(:nanosecond) - started_at, 0)
      cpu_after_ns = MLIR.CAPI.beaver_raw_process_cpu_time()
      rss_after = rss_bytes()
      rendered = MLIR.to_string(converted, generic: true)

      %{
        mode: if(profile?, do: "profiled", else: "baseline"),
        duration_ns: duration_ns,
        process_cpu_time_ns: max(cpu_after_ns - cpu_before_ns, 0),
        rss_before_bytes: rss_before,
        rss_after_bytes: rss_after,
        fingerprint: sha256(rendered),
        receipt: receipt
      }
    after
      MLIR.Context.destroy(ctx)
    end
  end

  defp summary(measurements) do
    durations = Enum.map(measurements, & &1.duration_ns)
    cpu_times = Enum.map(measurements, & &1.process_cpu_time_ns)
    rss_deltas = Enum.map(measurements, &max(&1.rss_after_bytes - &1.rss_before_bytes, 0))

    %{
      "duration_ns" => %{
        "median" => median(durations),
        "minimum" => Enum.min(durations),
        "maximum" => Enum.max(durations)
      },
      "process_cpu_time_ns" => %{
        "median" => median(cpu_times),
        "minimum" => Enum.min(cpu_times),
        "maximum" => Enum.max(cpu_times)
      },
      "rss_delta_bytes" => %{
        "median" => median(rss_deltas),
        "maximum" => Enum.max(rss_deltas)
      }
    }
  end

  defp assembly(functions) do
    body =
      0..(functions - 1)
      |> Enum.map_join("\n", fn index ->
        ~s"""
        "ex.func"() ({
        ^bb0:
          %0 = "ex.lit"() {value = #{index} : i64} : () -> i64
          %1 = "ex.box"(%0) : (i64) -> !ex.term
          "ex.return"(%1) {operandSegmentSizes = array<i32: 1>} : (!ex.term) -> ()
        }) {sym_name = "f#{index}"} : () -> ()
        """
      end)

    "module {\n#{body}\n}"
  end

  defp rss_bytes do
    case System.cmd("ps", ["-o", "rss=", "-p", System.pid()], stderr_to_stdout: true) do
      {output, 0} -> output |> String.trim() |> String.to_integer() |> Kernel.*(1024)
      {_output, _status} -> 0
    end
  end

  defp median(values) do
    sorted = Enum.sort(values)
    Enum.at(sorted, div(length(sorted), 2))
  end

  defp ratio(_numerator, 0), do: 0.0
  defp ratio(numerator, denominator), do: numerator / denominator

  defp paired_overhead_ratio(measurements) do
    measurements
    |> Enum.chunk_every(2)
    |> Enum.map(fn [
                     %{mode: "baseline", duration_ns: baseline_ns},
                     %{mode: "profiled", duration_ns: profiled_ns}
                   ] ->
      ratio(profiled_ns - baseline_ns, baseline_ns)
    end)
    |> median()
  end

  defp sha256(value), do: :crypto.hash(:sha256, value) |> Base.encode16(case: :lower)

  defp assert_success!(result, label) do
    unless MLIR.LogicalResult.success?(result), do: raise("failed while #{label}")
  end

  defp positive!(value, _name) when is_integer(value) and value > 0, do: value
  defp positive!(_value, name), do: raise(ArgumentError, "#{name} must be positive")

  defp non_negative!(value, _name) when is_integer(value) and value >= 0, do: value
  defp non_negative!(_value, name), do: raise(ArgumentError, "#{name} must be non-negative")

  defp optional_positive!(nil, _name), do: nil
  defp optional_positive!(value, _name) when is_number(value) and value > 0, do: value
  defp optional_positive!(_value, name), do: raise(ArgumentError, "#{name} must be positive")
end

Beaver.Profile.ExConversion.run(System.argv())
