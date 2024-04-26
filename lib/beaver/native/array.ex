defmodule Beaver.Native.Array do
  @moduledoc """
  This module defines functions working with C array.
  """
  alias Beaver.MLIR.CAPI
  defstruct ref: nil, element_kind: nil

  def as_opaque(%{ref: ref, element_kind: element_kind}) do
    ref =
      apply(CAPI, Module.concat([element_kind, "array_as_opaque"]), [
        ref
      ])
      |> Beaver.Native.check!()

    %Beaver.Native.OpaqueArray{ref: ref}
  end
end
