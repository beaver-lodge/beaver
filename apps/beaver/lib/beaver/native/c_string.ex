defmodule Beaver.Native.C.String do
  defstruct ref: nil, bag: MapSet.new(), element_module: Beaver.Native.U8

  def as_u8_array(%__MODULE__{ref: ref}) do
    %Beaver.Native.Array{
      ref: ref,
      element_module: Beaver.Native.U8
    }
  end
end
