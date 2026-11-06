defmodule BufferizationTest do
  use Beaver.Case, async: true

  alias Beaver.Composer
  alias Beaver.MLIR

  @moduletag :smoke

  @tensor_module """
  module {
    func.func @main() -> f32 {
      %tensor = bufferization.alloc_tensor() : tensor<4xf32>
      %c0 = arith.constant 0 : index
      %value = tensor.extract %tensor[%c0] : tensor<4xf32>
      func.return %value : f32
    }
  }
  """

  test "runs bufferization and preserves structured diagnostics", %{ctx: ctx} do
    module = MLIR.Module.create!(@tensor_module, ctx: ctx)

    assert {:ok, ^module, diagnostics} =
             module |> Composer.append("one-shot-bufferize") |> Composer.run()

    assert is_list(diagnostics)
    MLIR.verify!(module)
    assert to_string(module) =~ "memref.alloc"
    refute to_string(module) =~ "bufferization.alloc_tensor"
  end

  test "optionally appends ownership-based deallocation", %{ctx: ctx} do
    module = MLIR.Module.create!(@tensor_module, ctx: ctx)

    assert ^module =
             module
             |> Composer.append("one-shot-bufferize,buffer-deallocation-pipeline")
             |> Composer.run!()

    MLIR.verify!(module)
    assert to_string(module) =~ "memref.dealloc"
  end
end
