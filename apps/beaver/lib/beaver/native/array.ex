defmodule Beaver.Native.Array do
  defstruct ref: nil, element_module: nil, bag: MapSet.new()
end
