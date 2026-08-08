defmodule Beaver.Shadow.ProbeTest do
  use ExUnit.Case, async: true

  alias Beaver.Shadow.Probe

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
end
