defmodule Beaver.MLIR.Attribute do
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def get(attr) when is_binary(attr) do
    ctx = MLIR.Managed.Context.get()
    attr = MLIR.StringRef.create(attr)
    CAPI.mlirAttributeParseGet(ctx, attr)
  end
end
