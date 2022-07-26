defmodule Beaver.Native.Ptr do
  defstruct ref: nil, element_module: nil, bag: MapSet.new()
end
