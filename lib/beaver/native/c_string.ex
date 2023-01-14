defmodule Beaver.Native.C.String do
  @moduledoc """
  This module defines functions working with a C string, an array terminated with with NULL.
  """
  defstruct ref: nil, bag: MapSet.new(), element_kind: Beaver.Native.U8

  def as_u8_array(%__MODULE__{ref: ref}) do
    %Beaver.Native.Array{
      ref: ref,
      element_kind: Beaver.Native.U8
    }
  end
end
