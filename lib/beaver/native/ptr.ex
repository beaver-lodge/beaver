defmodule Beaver.Native.Ptr do
  defstruct ref: nil, element_kind: nil, bag: MapSet.new()
end
