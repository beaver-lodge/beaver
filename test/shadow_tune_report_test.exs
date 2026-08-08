defmodule Beaver.Shadow.TuneReportTest do
  use ExUnit.Case, async: true

  alias Beaver.Shadow.Probe.Result
  alias Beaver.Shadow.TuneReport
  alias Beaver.Shadow.Tuning
  alias Beaver.Shadow.Tuning.{Config, Record, Run}

  @triton_enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  defp sample_run do
    configs = [
      %Config{index: 0, num_warps: 2},
      %Config{index: 1, num_warps: 4},
      %Config{index: 2, num_warps: 8}
    ]

    record = fn config, optimized, status ->
      %Record{
        format: Tuning.format(),
        kernel_digest: String.duplicate("a", 64),
        target: "cuda:80",
        capability: nil,
        config_space_digest: Tuning.configs_digest(configs),
        config: config,
        status: status,
        failure: nil,
        structural_proxy: %{
          baseline: 24,
          optimized: optimized,
          reduction: 24 - optimized,
          lowered_to_llvm: true
        },
        timings: %{total_ns: 1_000_000}
      }
    end

    records = [
      record.(Enum.at(configs, 0), 5, :evaluated),
      record.(Enum.at(configs, 1), 5, :evaluated),
      record.(Enum.at(configs, 2), 4, :evaluated)
    ]

    %Run{
      fixture: :matmul,
      kernel_digest: String.duplicate("a", 64),
      records: records,
      winner: List.last(records)
    }
  end

  describe "render/1" do
    test "includes environment facts, records, winner, and probes" do
      run = sample_run()

      report = %TuneReport{
        fixture: :matmul,
        generated_at: ~U[2026-08-08 00:00:00Z],
        llvm_revision: "LLVM version 24.0.0git",
        prebuilt_dir: "/tmp/prebuilt",
        target: "cuda:80",
        config_space_digest: Tuning.configs_digest(run.records |> Enum.map(& &1.config)),
        records: run.records,
        winner: run.winner,
        probes: [
          %{
            config: %Config{index: 0, num_warps: 2},
            result: %Result{fixture: :matmul, status: :ok, detail: %{llvm_func: true}}
          },
          %{
            config: %Config{index: 1, num_warps: 4},
            result: %Result{
              fixture: :matmul,
              status: :crash,
              exit_code: 139,
              detail: %{last_pass: "tritongpu-accelerate-matmul"}
            }
          }
        ]
      }

      rendered = TuneReport.render(report)

      assert rendered =~ "llvm revision: LLVM version 24.0.0git"
      assert rendered =~ "triton prebuilt dir: `/tmp/prebuilt`"
      assert rendered =~ "| 0 | 2 | - | - |"
      assert rendered =~ "| 2 | evaluated | 24 | 4 | -20 | true | 1000000 |"
      assert rendered =~ "### Winner" and rendered =~ "num_warps 8"
      assert rendered =~ "| 0 | 2 | ok |"
      assert rendered =~ "| 1 | 4 | crash (exit 139) at `tritongpu-accelerate-matmul` |"
    end
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "generate/0 measures a config space and probes on the corpus" do
    configs = [
      %Config{index: 0, num_warps: 2},
      %Config{index: 1, num_warps: 4},
      %Config{index: 2, num_warps: 8}
    ]

    report = TuneReport.generate(:matmul, configs)

    assert report.llvm_revision != nil
    assert length(report.records) == 3
    assert report.winner.status == :evaluated
    assert length(report.probes) == 3
    assert Enum.all?(report.probes, &(&1.result.status == :ok))

    rendered = TuneReport.render(report)
    assert rendered =~ "## Shadow Wavefront: Triton tuning measurement"
    assert rendered =~ "| 0 | evaluated | 24 | 5 | -19 |"
  end
end
