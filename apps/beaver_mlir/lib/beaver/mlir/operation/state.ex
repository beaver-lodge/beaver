defmodule Beaver.MLIR.Operation.State do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI.IR
  alias Exotic.Value.Array
  alias Exotic.Value
  defstruct [:ref]

  def get!(name, location) when is_binary(name) do
    IR.mlirOperationStateGet(
      IR.string_ref(name),
      location
    )
  end

  def get!(context, name) when is_binary(name) do
    get!(name, IR.mlirLocationUnknownGet(context))
  end

  defp get_context(state) do
    location = Exotic.Value.fetch(state, IR.OperationState, :location)
    IR.mlirLocationGetContext(location)
  end

  def add_attr(state, attrs) when is_list(attrs) do
    ctx = get_context(state)

    named_attrs =
      for {k, v} <- attrs do
        k = Atom.to_string(k)

        IR.mlirNamedAttributeGet(
          IR.mlirIdentifierGet(ctx, IR.string_ref(k)),
          IR.mlirAttributeParseGet(ctx, IR.string_ref(v))
          |> Exotic.Value.transmit()
        )
      end

    MLIR.Operation.State.add_attrs(state, named_attrs)
    state
  end

  def add_operand(state, operands) do
    array =
      operands
      |> Enum.map(&Exotic.Value.transmit/1)
      |> Array.get()
      |> Value.get_ptr()

    IR.mlirOperationStateAddOperands(Value.get_ptr(state), length(operands), array)
    state
  end

  def add_result(state, result_types) do
    context = get_context(state)

    array =
      result_types
      |> Enum.map(fn t ->
        IR.mlirTypeParseGet(context, IR.string_ref(t))
      end)
      |> Enum.map(&Exotic.Value.transmit/1)
      |> Array.get()
      |> Value.get_ptr()

    IR.mlirOperationStateAddResults(Exotic.Value.get_ptr(state), length(result_types), array)
    state
  end

  def add_attrs(state, attrs) when is_list(attrs) do
    array_ptr = attrs |> Enum.map(&Value.transmit/1) |> Array.get() |> Value.get_ptr()

    IR.mlirOperationStateAddAttributes(
      Value.get_ptr(state),
      length(attrs),
      array_ptr
    )
  end

  def add_regions(state, regions) when is_list(regions) do
    array_ptr = regions |> Enum.map(&Value.transmit/1) |> Array.get() |> Value.get_ptr()

    IR.mlirOperationStateAddOwnedRegions(
      Value.get_ptr(state),
      length(regions),
      array_ptr
    )

    state
  end
end
