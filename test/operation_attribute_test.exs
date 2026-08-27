defmodule OperationAttributeTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  test "keeps inherent properties and discardable metadata on separate paths", %{ctx: ctx} do
    module = MLIR.Module.create!("module { func.func @example() { return } }", ctx: ctx)
    operation = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)

    assert MLIR.Operation.inherent_attribute?(operation, :sym_name)
    assert MLIR.Operation.inherent_attribute?(operation, :sym_visibility)

    assert MLIR.Operation.inherent_attribute(operation, :sym_name) |> to_string() ==
             ~s("example")

    assert MLIR.Operation.inherent_attribute(operation, :sym_visibility) == nil
    refute MLIR.Operation.inherent_attribute?(operation, "test.note")

    note = MLIR.Attribute.string("metadata", ctx: ctx)
    assert :ok = MLIR.Operation.put_discardable_attribute(operation, "test.note", note)
    assert MLIR.equal?(MLIR.Operation.discardable_attribute(operation, "test.note"), note)
    assert MLIR.equal?(MLIR.Operation.attribute(operation, "test.note"), note)

    inherent_names = attribute_names(Beaver.Walker.inherent_attributes(operation))
    discardable_names = attribute_names(Beaver.Walker.discardable_attributes(operation))

    assert :sym_name in inherent_names
    refute :sym_name in discardable_names
    assert :"test.note" in discardable_names

    assert Enum.sort(attribute_names(Beaver.Walker.attributes(operation))) ==
             Enum.sort(inherent_names ++ discardable_names)

    assert_raise ArgumentError, ~r/:sym_name is inherent/, fn ->
      MLIR.Operation.put_discardable_attribute(operation, :sym_name, note)
    end
  end

  test "Access dispatches using the operation schema", %{ctx: ctx} do
    module = MLIR.Module.create!("module { func.func @example() { return } }", ctx: ctx)
    operation = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)

    private = MLIR.Attribute.string("private", ctx: ctx)
    tagged = MLIR.Attribute.unit(ctx: ctx)

    assert ^operation = put_in(operation[:sym_visibility], private)
    assert to_string(operation[:sym_visibility]) == ~s("private")

    assert ^operation = put_in(operation["test.tag"], tagged)
    assert MLIR.equal?(operation["test.tag"], tagged)

    {popped, ^operation} = pop_in(operation["test.tag"])
    assert MLIR.equal?(popped, tagged)
    assert operation["test.tag"] == nil
  end

  test "treats attributes on unregistered operations as discardable", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)

    module =
      MLIR.Module.create!(
        ~S[module { "test.unknown"() {test.value = 1 : i32} : () -> () }],
        ctx: ctx
      )

    operation = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)

    refute MLIR.Operation.inherent_attribute?(operation, "test.value")

    assert MLIR.Operation.discardable_attribute(operation, "test.value") |> to_string() ==
             "1 : i32"

    assert attribute_names(Beaver.Walker.inherent_attributes(operation)) == []
    assert attribute_names(Beaver.Walker.discardable_attributes(operation)) == [:"test.value"]
  end

  test "treats attributes on property-less dynamic operations as discardable", %{ctx: ctx} do
    assert CompleteSlang |> then(&Beaver.Slang.load(ctx, &1)) |> MLIR.LogicalResult.success?()

    module =
      MLIR.Module.create!(
        ~S[module { "complete_slang.scope"() ({ ^bb0: }) {label = "dynamic"} : () -> () }],
        ctx: ctx
      )

    operation = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.fetch!(0)

    refute MLIR.Operation.inherent_attribute?(operation, :label)
    assert MLIR.Operation.inherent_attribute(operation, :label) == nil
    assert MLIR.Operation.discardable_attribute(operation, :label) |> to_string() == ~s("dynamic")
  end

  defp attribute_names(attributes) do
    Enum.map(attributes, fn {identifier, _attribute} ->
      identifier |> MLIR.CAPI.mlirIdentifierStr() |> MLIR.to_string() |> String.to_atom()
    end)
  end
end
