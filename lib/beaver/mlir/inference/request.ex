defmodule Beaver.MLIR.Inference.Request do
  @moduledoc false

  alias Beaver.MLIR

  @spec raw_arguments(map() | keyword()) :: [term()]
  def raw_arguments(request) when is_map(request) or is_list(request) do
    request = Map.new(request)
    context = Map.fetch!(request, :context)
    unless match?(%MLIR.Context{}, context), do: raise(ArgumentError, ":context must be a Beaver.MLIR.Context")
    operation = normalize_operation(Map.fetch!(request, :operation))
    location = Map.get_lazy(request, :location, fn -> MLIR.Location.unknown(ctx: context) end)
    operands = Map.get(request, :operands, [])
    regions = Map.get(request, :regions, [])

    attributes = Map.get_lazy(request, :attributes, fn -> MLIR.Attribute.dictionary([], ctx: context) end)

    properties = properties_ref(Map.get(request, :properties))

    unless match?(%MLIR.Location{}, location), do: raise(ArgumentError, ":location must be a Beaver.MLIR.Location")

    unless match?(%MLIR.Attribute{}, attributes),
      do: raise(ArgumentError, ":attributes must be a dictionary Beaver.MLIR.Attribute")

    unless Enum.all?(operands, &match?(%MLIR.Value{}, &1)),
      do: raise(ArgumentError, ":operands must contain only Beaver.MLIR.Value entries")

    unless Enum.all?(regions, &match?(%MLIR.Region{}, &1)),
      do: raise(ArgumentError, ":regions must contain only Beaver.MLIR.Region entries")

    [
      operation,
      context.ref,
      location.ref,
      Enum.map(operands, & &1.ref),
      attributes.ref,
      properties,
      Enum.map(regions, & &1.ref)
    ]
  end

  defp properties_ref(nil), do: nil
  defp properties_ref(%MLIR.Attribute{ref: ref}), do: ref
  defp properties_ref(_), do: raise(ArgumentError, ":properties must be a Beaver.MLIR.Attribute or nil")

  defp normalize_operation(op) when is_binary(op), do: op
  defp normalize_operation(op) when is_atom(op), do: Atom.to_string(op)

  defp normalize_operation(op),
    do: raise(ArgumentError, ":operation must be a canonical operation name, got: #{inspect(op)}")
end
