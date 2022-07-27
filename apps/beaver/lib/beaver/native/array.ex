defmodule Beaver.Native.Array do
  alias Beaver.MLIR.CAPI
  defstruct ref: nil, element_module: nil, bag: MapSet.new()

  def as_opaque(%{ref: ref, element_module: element_module}) do
    ref =
      apply(CAPI, Module.concat([element_module, "array_as_opaque"]), [
        ref
      ])
      |> Beaver.Native.check!()

    %Beaver.Native.OpaqueArray{ref: ref}
  end
end
