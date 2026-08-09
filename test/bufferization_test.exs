defmodule BufferizationTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Bufferization

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

  test "renders structured One-Shot Bufferize options" do
    pipeline =
      Bufferization.one_shot_pipeline(
        bufferize_function_boundaries: true,
        boundary_layout: :identity,
        allow_unknown_ops: true,
        analysis_heuristic: :bottom_up_from_terminators
      )

    assert pipeline =~ "bufferize-function-boundaries=true"
    assert pipeline =~ "unknown-type-conversion=identity-layout-map"
    assert pipeline =~ "function-boundary-type-conversion=identity-layout-map"
    assert pipeline =~ "analysis-heuristic=bottom-up-from-terminators"
  end

  test "runs bufferization and preserves structured diagnostics", %{ctx: ctx} do
    module = MLIR.Module.create!(@tensor_module, ctx: ctx)

    assert {:ok, ^module, diagnostics} = Bufferization.one_shot(module)
    assert is_list(diagnostics)
    MLIR.verify!(module)
    assert to_string(module) =~ "memref.alloc"
    refute to_string(module) =~ "bufferization.alloc_tensor"
  end

  test "optionally appends ownership-based deallocation", %{ctx: ctx} do
    module = MLIR.Module.create!(@tensor_module, ctx: ctx)

    assert ^module = Bufferization.one_shot!(module, deallocate: true)
    MLIR.verify!(module)
    assert to_string(module) =~ "memref.dealloc"
  end

  test "rejects unknown and invalid options" do
    assert_raise ArgumentError, ~r/unsupported bufferization option/, fn ->
      Bufferization.one_shot_pipeline(unknown: true)
    end

    assert_raise ArgumentError, ~r/allow_unknown_ops must be a boolean/, fn ->
      Bufferization.one_shot_pipeline(allow_unknown_ops: :yes)
    end
  end
end
