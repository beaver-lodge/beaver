defmodule Beaver.MLIR.Value do
  alias Beaver.MLIR.CAPI

  def dump(value) do
    value |> CAPI.mlirValueDump()
  end
end
