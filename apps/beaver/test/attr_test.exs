defmodule MLIRAttrTest do
  use ExUnit.Case

  test "spv attributes" do
    import Beaver.MLIR.Sigils
    ~a{#spv.entry_point_abi<local_size = dense<1> : vector<3xi32>>}

    ~a{#spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spv.resource_limits<>>}
  end
end
