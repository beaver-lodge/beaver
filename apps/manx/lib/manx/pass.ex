defmodule Manx.Lowering.Vulkan.PutSPVAttrPass do
  alias Beaver.MLIR
  import MLIR.Sigils
  alias Beaver.MLIR.Dialect.GPU
  use Beaver.MLIR.Pass, on: GPU.Func

  def run(op) do
    [
      "gpu.kernel": Beaver.MLIR.Attribute.unit(),
      "spv.entry_point_abi":
        ~a{#spv.entry_point_abi<local_size = dense<[16, 1, 1]> : vector<3xi32>>}
    ]
    |> Enum.each(fn {n, a} ->
      MLIR.CAPI.mlirOperationSetAttributeByName(op, MLIR.StringRef.create(n), a)
    end)

    :ok
  end
end
