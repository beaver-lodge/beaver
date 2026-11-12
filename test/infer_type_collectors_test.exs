defmodule Beaver.MLIR.InferTypeCollectorsTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.{InferShapedType, InferType}

  @moduletag :smoke

  defp with_args(ctx, types, fun) do
    Beaver.Native.with_resource(MLIR.Block.create(), &MLIR.Block.destroy/1, fn block ->
      fun.(MLIR.Block.add_args!(block, types, ctx: ctx))
    end)
  end

  test "collects single and multiple InferType results", %{ctx: ctx} do
    with_args(ctx, ["i32", "i32"], fn operands ->
      for {operation, count} <- [{"arith.addi", 1}, {"arith.mului_extended", 2}] do
        assert {:ok, types} = InferType.return_types(operation: operation, context: ctx, operands: operands)

        assert Enum.map(types, &MLIR.to_string/1) == List.duplicate("i32", count)
      end
    end)
  end

  test "returns diagnostics when inference fails", %{ctx: ctx} do
    assert {:error, diagnostics} = InferType.return_types(operation: "not.registered", context: ctx)

    assert diagnostics != []
    assert MLIR.Diagnostic.format(diagnostics) =~ "failed to infer operation return types"
  end

  test "collects ranked dynamic and unranked shaped components", %{ctx: ctx} do
    for {type, shape} <- [{"tensor<2x?xf32>", [2, :dynamic]}, {"tensor<*xf32>", :unranked}] do
      with_args(ctx, [type], fn [input] ->
        assert {:ok, [%InferShapedType.Component{shape: ^shape, element_type: element_type, encoding: nil}]} =
                 InferShapedType.return_components(operation: "tosa.equal", context: ctx, operands: [input, input])

        assert MLIR.to_string(element_type) == "i1"
      end)
    end
  end

  test "rejects untyped operation properties explicitly", %{ctx: ctx} do
    assert_raise ArgumentError, ~r/:properties must be a Beaver.MLIR.Attribute/, fn ->
      InferType.return_types(operation: "arith.addi", context: ctx, properties: %{})
    end
  end

  test "infers from owned property storage without constructing or mutating an operation", %{ctx: ctx} do
    properties = MLIR.Attribute.get("{value = 7 : i32}", ctx: ctx)

    for _ <- 1..20 do
      assert {:ok, [type]} = InferType.return_types(operation: "arith.constant", context: ctx, properties: properties)

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

    assert {:error, _} = InferType.return_types(operation: "func.func", context: ctx, properties: properties)
  end

  test "rejects inference attributes from another context", %{ctx: ctx} do
    Beaver.Native.with_resource(MLIR.Context.create(), &MLIR.Context.destroy/1, fn other ->
      assert {:error, _} =
               InferType.return_types(
                 operation: "arith.constant",
                 context: ctx,
                 properties: MLIR.Attribute.get("{value = 7 : i32}", ctx: other)
               )
    end)
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

    assert {:ok, [type]} = InferType.return_types(operation: "scf.if", context: ctx, operands: [cond], regions: regions)

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
