defmodule Beaver.Shadow.ProbeTest do
  use ExUnit.Case, async: true

  alias Beaver.Shadow.Probe
  alias Beaver.Shadow.Tuning

  @triton_enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  @tag :triton
  @tag skip: !@triton_enabled
  test "TTIR fixtures lower to LLVM through the probe" do
    for fixture <- [:matmul, :attention] do
      result = Probe.run(fixture)
      assert result.status == :ok, "#{fixture}: #{inspect(result)}"
      assert result.detail.llvm_func
    end
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "remat TTGIR probe records the pinned prebuilt crash deterministically" do
    result = Probe.run(:remat)

    # Pinned prebuilt fact: tritongpu-accelerate-matmul segfaults on a
    # dot-free TTGIR module, so lowering the remat slice crashes the child
    # BEAM. This assertion flips to :ok once the prebuilt is fixed or bumped.
    assert result.status == :crash
    assert result.exit_code != 0
    assert result.detail.last_pass == "tritongpu-accelerate-matmul"
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "probe_many isolates every config in its own child BEAM" do
    configs = [
      %Tuning.Config{index: 0, num_warps: 2},
      %Tuning.Config{index: 1, num_warps: 4},
      %Tuning.Config{index: 2, num_warps: 8}
    ]

    results = Probe.probe_many(:matmul, configs)

    assert length(results) == 3

    for %{config: config, result: result} <- results do
      assert result.status == :ok, "config #{config.num_warps}: #{inspect(result)}"
      assert result.detail.llvm_func
    end
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "probe_many records per-config failure receipts for the remat fixture" do
    configs = [
      %Tuning.Config{index: 0, num_warps: 2},
      %Tuning.Config{index: 1, num_warps: 4}
    ]

    results = Probe.probe_many(:remat, configs)

    assert length(results) == 2

    for %{config: config, result: result} <- results do
      assert result.status == :crash, "config #{config.num_warps}: #{inspect(result)}"
      assert result.exit_code != 0
      assert result.detail.last_pass == "tritongpu-accelerate-matmul"
    end
  end
end
