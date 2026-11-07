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
    assert_raise ArgumentError, ~r/:properties must be a Beaver.MLIR.Attribute/, fn ->
      InferType.return_types(operation: "arith.addi", context: ctx, properties: %{})
    end
  end

  test "infers from owned property storage without constructing or mutating an operation", %{
    ctx: ctx
  } do
    properties = MLIR.Attribute.get("{value = 7 : i32}", ctx: ctx)

    for _ <- 1..20 do
      assert {:ok, [type]} =
               InferType.return_types(
                 operation: "arith.constant",
                 context: ctx,
                 properties: properties
               )

      assert MLIR.to_string(type) == "i32"
    end

    assert MLIR.to_string(properties) == "{value = 7 : i32}"

    assert {:error, diagnostics} =
             InferType.return_types(
               operation: "arith.constant",
               context: ctx,
               properties: MLIR.Attribute.get("42 : i32", ctx: ctx)
             )

    assert MLIR.Diagnostic.format(diagnostics) =~ "expected DictionaryAttr"

    assert {:error, _} =
             InferType.return_types(operation: "func.func", context: ctx, properties: properties)
  end

  test "rejects inference attributes from another context", %{ctx: ctx} do
    other = MLIR.Context.create()

    try do
      assert {:error, _} =
               InferType.return_types(
                 operation: "arith.constant",
                 context: ctx,
                 properties: MLIR.Attribute.get("{value = 7 : i32}", ctx: other)
               )
    after
      MLIR.Context.destroy(other)
    end
  end

  test "shaped inference uses properties and borrows regions without consuming them", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        module {
          func.func @infer(%input: tensor<2x3xf32>, %cond: i1) -> i32 {
            %result = scf.if %cond -> (i32) {
              %a = arith.constant 1 : i32
              scf.yield %a : i32
            } else {
              %b = arith.constant 2 : i32
              scf.yield %b : i32
            }
            return %result : i32
          }
        }
        """,
        ctx: ctx
      )

    func = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)
    block = func |> Beaver.Walker.regions() |> Enum.at(0) |> Beaver.Walker.blocks() |> Enum.at(0)
    input = MLIR.Block.get_arg!(block, 0)
    cond = MLIR.Block.get_arg!(block, 1)

    assert {:ok, [%InferShapedType.Component{shape: [2, 6]}]} =
             InferShapedType.return_components(
               operation: "tosa.concat",
               context: ctx,
               operands: [input, input],
               properties: MLIR.Attribute.get("{axis = 1 : i32}", ctx: ctx)
             )

    branch = block |> Beaver.Walker.operations() |> Enum.at(0)
    regions = Enum.to_list(Beaver.Walker.regions(branch))
    before = MLIR.to_string(module)

    assert {:ok, [type]} =
             InferType.return_types(
               operation: "scf.if",
               context: ctx,
               operands: [cond],
               regions: regions
             )

    assert MLIR.to_string(type) == "i32"
    assert MLIR.to_string(module) == before
    MLIR.verify!(module)

    assert {:error, _} =
             InferType.return_types(
               operation: "arith.constant",
               context: ctx,
               properties: MLIR.Attribute.get("{value = []}", ctx: ctx)
             )

    MLIR.Module.destroy(module)
  end
end
