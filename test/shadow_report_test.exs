defmodule Beaver.Shadow.ReportTest do
  use ExUnit.Case, async: true

  alias Beaver.Shadow.Probe.Result
  alias Beaver.Shadow.Report

  @triton_enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  describe "render/1" do
    test "renders a markdown table from collected facts" do
      report = %Report{
        generated_at: ~U[2026-08-08 00:00:00Z],
        llvm_revision: "LLVM version 20.1.0",
        prebuilt_dir: "/tmp/prebuilt",
        fixtures: [
          %{
            name: :matmul,
            dialect: :ttir,
            baseline: 24,
            optimized: 5,
            lowered_to_llvm: true,
            probe: nil
          },
          %{
            name: :remat,
            dialect: :ttgir,
            baseline: 4,
            optimized: 1,
            lowered_to_llvm: false,
            probe: %Result{
              fixture: :remat,
              status: :crash,
              exit_code: 139,
              detail: %{last_pass: "tritongpu-accelerate-matmul"}
            }
          }
        ]
      }

      rendered = Report.render(report)

      assert rendered =~ "| matmul | ttir | 24 | 5 | -19 | yes | - |"

      assert rendered =~
               "| remat | ttgir | 4 | 1 | -3 | no | crash (exit 139) at `tritongpu-accelerate-matmul` |"

      assert rendered =~ "llvm revision: LLVM version 20.1.0"
      assert rendered =~ "triton prebuilt dir: `/tmp/prebuilt`"
    end
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "generate/0 measures the whole corpus" do
    report = Report.generate()

    assert is_binary(report.llvm_revision) and report.llvm_revision != ""
    assert length(report.fixtures) == 4

    matmul = Enum.find(report.fixtures, &(&1.name == :matmul))
    assert matmul.baseline == 24
    assert matmul.optimized == 5
    assert matmul.lowered_to_llvm

    attention = Enum.find(report.fixtures, &(&1.name == :attention))
    assert attention.baseline == 34
    assert attention.optimized == 5

    remat = Enum.find(report.fixtures, &(&1.name == :remat))
    assert remat.baseline == 4
    assert remat.optimized == 1
    assert remat.probe.status == :crash
  end
end
