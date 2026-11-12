defmodule OperationAttributeTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR.{Attribute, Bytecode, Context, LogicalResult, Operation}

  alias Beaver.MLIR

  test "keeps inherent properties and discardable metadata on separate paths", %{ctx: ctx} do
    module = MLIR.Module.create!("module { func.func @example() { return } }", ctx: ctx)
    operation = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)
    assert Operation.inherent_attribute?(operation, :sym_name)
    assert Operation.inherent_attribute?(operation, :sym_visibility)

    assert Operation.inherent_attribute(operation, :sym_name) |> to_string() == ~s("example")

    assert Operation.inherent_attribute(operation, :sym_visibility) == nil
    refute Operation.inherent_attribute?(operation, "test.note")
    note = Attribute.string("metadata", ctx: ctx)
    assert :ok = Operation.put_discardable_attribute(operation, "test.note", note)
    assert MLIR.equal?(Operation.discardable_attribute(operation, "test.note"), note)
    assert MLIR.equal?(Operation.attribute(operation, "test.note"), note)
    inherent_names = attribute_names(Beaver.Walker.inherent_attributes(operation))
    discardable_names = attribute_names(Beaver.Walker.discardable_attributes(operation))
    assert :sym_name in inherent_names
    refute :sym_name in discardable_names
    assert :"test.note" in discardable_names

    assert Enum.sort(attribute_names(Beaver.Walker.attributes(operation))) ==
             Enum.sort(inherent_names ++ discardable_names)

    assert_raise ArgumentError, ~r/:sym_name is inherent/, fn ->
      Operation.put_discardable_attribute(operation, :sym_name, note)
    end
  end

  test "Access dispatches using the operation schema", %{ctx: ctx} do
    module = MLIR.Module.create!("module { func.func @example() { return } }", ctx: ctx)
    operation = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)
    private = Attribute.string("private", ctx: ctx)
    tagged = Attribute.unit(ctx: ctx)
    assert ^operation = put_in(operation[:sym_visibility], private)
    assert to_string(operation[:sym_visibility]) == ~s("private")
    assert ^operation = put_in(operation["test.tag"], tagged)
    assert MLIR.equal?(operation["test.tag"], tagged)
    {popped, ^operation} = pop_in(operation["test.tag"])
    assert MLIR.equal?(popped, tagged)
    assert operation["test.tag"] == nil
  end

  test "treats attributes on unregistered operations as discardable", %{ctx: ctx} do
    Context.allow_unregistered_dialects(ctx)

    module = MLIR.Module.create!(~S[module { "test.unknown"() {test.value = 1 : i32} : () -> () }], ctx: ctx)

    operation = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)
    refute Operation.inherent_attribute?(operation, "test.value")

    assert Operation.discardable_attribute(operation, "test.value") |> to_string() == "1 : i32"

    assert attribute_names(Beaver.Walker.inherent_attributes(operation)) == []
    assert attribute_names(Beaver.Walker.discardable_attributes(operation)) == [:"test.value"]
  end

  test "treats attributes on property-less dynamic operations as discardable", %{ctx: ctx} do
    assert CompleteSlang |> then(&Beaver.Slang.load(ctx, &1)) |> LogicalResult.success?()

    module =
      ~S[module { "complete_slang.scope"() ({ ^bb0: }) {label = "dynamic"} : () -> () }]
      |> MLIR.Module.create!(ctx: ctx)

    operation = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)
    refute Operation.inherent_attribute?(operation, :label)
    assert Operation.inherent_attribute(operation, :label) == nil
    assert Operation.discardable_attribute(operation, :label) |> to_string() == ~s("dynamic")
  end

  test "preserves attribute classes through generic assembly and bytecode", %{ctx: ctx} do
    original = MLIR.Module.create!("module { func.func @example() { return } }", ctx: ctx)
    original_function = operation_named(original, "func.func")

    Operation.put_discardable_attribute(original_function, "test.note", Attribute.string("round-trip", ctx: ctx))

    text_round_trip = original |> MLIR.to_string(generic: true) |> MLIR.Module.create!(ctx: ctx)

    bytecode_round_trip = original |> Bytecode.write!() |> Bytecode.read!(ctx: ctx)

    for round_trip <- [text_round_trip, bytecode_round_trip] do
      function = operation_named(round_trip, "func.func")
      assert Operation.inherent_attribute?(function, :sym_name)

      assert Operation.inherent_attribute(function, :sym_name) |> to_string() == ~s("example")

      assert Operation.discardable_attribute(function, "test.note") |> to_string() == ~s("round-trip")

      assert Operation.equivalent?(original_function, function, ignore_locations: true)

      assert Operation.structural_hash(original_function, ignore_locations: true) ==
               Operation.structural_hash(function, ignore_locations: true)
    end
  end

  test "round-trips custom property syntax without merging discardable metadata", %{ctx: ctx} do
    {module, keyed_properties?} =
      case MLIR.Module.create(emitc_source("<args = [i32]>"), ctx: ctx) do
        {:ok, module} -> {module, true}
        {:error, _diagnostics} -> {MLIR.Module.create!(emitc_source("<{args = [i32]}>"), ctx: ctx), false}
      end

    rendered = MLIR.to_string(module)

    if keyed_properties? do
      assert rendered =~ ~s|emitc.call_opaque "sizeof"() <args = [i32]> {test.note}|
    else
      assert rendered =~ ~s|emitc.call_opaque "sizeof"() <{args = [i32]}> {test.note}|
    end

    for source <- [MLIR.to_string(module, generic: true), Bytecode.write!(module)] do
      round_trip = MLIR.Module.create!(source, ctx: ctx)
      call = operation_named(round_trip, "emitc.call_opaque")
      assert Operation.inherent_attribute?(call, :args)
      assert Operation.inherent_attribute(call, :args) |> to_string() == "[i32]"
      assert Operation.discardable_attribute(call, "test.note") |> to_string() == "unit"
    end
  end

  test "keeps alloc_tensor properties separate as its upstream assembly evolves", %{ctx: ctx} do
    strict_source = alloc_tensor_source("<{memory_space = 1 : i64}> {test.escape}")

    {module, strict_properties?} =
      case MLIR.Module.create(strict_source, ctx: ctx) do
        {:ok, module} ->
          {module, true}

        {:error, _diagnostics} ->
          {MLIR.Module.create!(alloc_tensor_source("{memory_space = 1 : i64, test.escape}"), ctx: ctx), false}
      end

    alloc = operation_named(module, "bufferization.alloc_tensor")
    assert Operation.inherent_attribute?(alloc, :memory_space)
    assert Operation.inherent_attribute(alloc, :memory_space) |> to_string() == "1 : i64"
    assert Operation.discardable_attribute(alloc, "test.escape") |> to_string() == "unit"

    if strict_properties? do
      assert MLIR.to_string(module) =~ "<{memory_space = 1 : i64}> {test.escape}"
    end

    bytecode_round_trip = module |> Bytecode.write!() |> Bytecode.read!(ctx: ctx)
    round_trip_alloc = operation_named(bytecode_round_trip, "bufferization.alloc_tensor")
    assert Operation.inherent_attribute?(round_trip_alloc, :memory_space)

    assert Operation.discardable_attribute(round_trip_alloc, "test.escape") |> to_string() == "unit"
  end

  defp attribute_names(attributes) do
    Enum.map(attributes, fn {identifier, _attribute} ->
      identifier |> MLIR.CAPI.mlirIdentifierStr() |> MLIR.to_string() |> String.to_atom()
    end)
  end

  defp operation_named(module, name), do: module |> Beaver.Walker.find_ops(name) |> List.first()

  defp alloc_tensor_source(attributes) do
    "module {\n  func.func @allocate() -> tensor<4xf32> {\n    %0 = bufferization.alloc_tensor() #{attributes} : tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n}\n"
  end

  defp emitc_source(properties) do
    "module {\n  func.func @sizeof() -> !emitc.size_t {\n    %0 = emitc.call_opaque \"sizeof\"() #{properties} {test.note} : () -> !emitc.size_t\n    return %0 : !emitc.size_t\n  }\n}\n"
  end
end
