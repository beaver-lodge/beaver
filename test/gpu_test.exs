defmodule GPUTest do
  use Beaver.Case
  use Beaver
  alias Beaver.MLIR

  @moduletag :smoke

  test "multiple regions", test_context do
    ctx = test_context[:ctx]
    MLIR.CAPI.mlirRegisterAllLLVMTranslations(ctx)

    MLIR.Module.create(ctx, File.read!("test/gpu-to-cubin.mlir"))
    |> MLIR.Pass.Composer.append("gpu-lower-to-nvvm-pipeline{cubin-format=isa}")
    |> MLIR.Pass.Composer.run!()
    |> MLIR.dump!()
  end
end
