defmodule Beaver.MLIR.CAPI.Transform do
  @deprecated "Use Beaver.MLIR.CAPI"
  @moduledoc false
  use Exotic.Library
  @path "libMLIRTransform.dylib"
  def load!(), do: Exotic.load!(__MODULE__, Beaver.MLIR.CAPI)
end
