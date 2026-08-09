defmodule Beaver.MLIR.InferTypeCollectorsTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.Func
  alias Beaver.MLIR.{InferShapedType, InferType}

  require Func

  @moduletag :smoke

  test "collects single and multiple InferType results", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        Func.func infer(function_type: Type.function([Type.i32(), Type.i32()], [])) do
          region do
            block _entry(lhs >>> Type.i32(), rhs >>> Type.i32()) do
              assert {:ok, [single]} =
                       InferType.return_types(
                         operation: "arith.addi",
                         context: ctx,
                         operands: [lhs, rhs]
                       )

              assert MLIR.to_string(single) == "i32"

              assert {:ok, [low, high]} =
                       InferType.return_types(
                         operation: "arith.mului_extended",
                         context: ctx,
                         operands: [lhs, rhs]
                       )

              assert Enum.map([low, high], fn type -> MLIR.to_string(type) end) == ["i32", "i32"]
              Func.return() >>> []
            end
          end
        end
      end
    end
    |> MLIR.verify!()
  end

  test "returns diagnostics when inference fails", %{ctx: ctx} do
    assert {:error, diagnostics} =
             InferType.return_types(operation: "not.registered", context: ctx)

    assert diagnostics != []
    assert MLIR.Diagnostic.format(diagnostics) =~ "failed to infer operation return types"
  end

  test "collects ranked dynamic and unranked shaped components", %{ctx: ctx} do
    mlir ctx: ctx do
      module do
        ranked = ~t{tensor<2x?xf32>}
        unranked = ~t{tensor<*xf32>}

        Func.func infer_ranked(function_type: Type.function([ranked], [])) do
          region do
            block _entry(input >>> ranked) do
              assert {:ok, [component]} =
                       InferShapedType.return_components(
                         operation: "tosa.equal",
                         context: ctx,
                         operands: [input, input]
                       )

              assert %InferShapedType.Component{
                       shape: [2, :dynamic],
                       element_type: element_type,
                       encoding: nil
                     } = component

              assert MLIR.to_string(element_type) == "i1"
              Func.return() >>> []
            end
          end
        end

        Func.func infer_unranked(function_type: Type.function([unranked], [])) do
          region do
            block _entry(input >>> unranked) do
              assert {:ok, [%InferShapedType.Component{shape: :unranked}]} =
                       InferShapedType.return_components(
                         operation: "tosa.equal",
                         context: ctx,
                         operands: [input, input]
                       )

              Func.return() >>> []
            end
          end
        end
      end
    end
    |> MLIR.verify!()
  end

  test "rejects untyped operation properties explicitly", %{ctx: ctx} do
    assert_raise ArgumentError, ~r/currently support only properties: nil/, fn ->
      InferType.return_types(operation: "arith.addi", context: ctx, properties: %{})
    end
  end
end
