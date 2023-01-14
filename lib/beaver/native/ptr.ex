defmodule Beaver.Native.Ptr do
  @moduledoc """
  This module defines functions working with pointer in C.
  """
  defstruct ref: nil, element_kind: nil, bag: MapSet.new()
end
