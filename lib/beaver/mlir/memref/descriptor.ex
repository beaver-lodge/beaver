defmodule Beaver.MLIR.MemRef.Descriptor do
  @doc """
  Allocate memory base on the shape and element type information. A new MemRef descriptor resource is created and returned. The resource will own the allocated memory and will be automatically deallocated when the resource is garbage collected.
  """
  def allocate() do
  end
end
