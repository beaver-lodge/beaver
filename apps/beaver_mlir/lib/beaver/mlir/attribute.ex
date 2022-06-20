defmodule Beaver.MLIR.Attribute do
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def is_null(jit) do
    jit
    |> Exotic.Value.fetch(MLIR.CAPI.MlirAttribute, :ptr)
    |> Exotic.Value.extract() == 0
  end

  def get(attr_str) when is_binary(attr_str) do
    ctx = MLIR.Managed.Context.get()
    attr = MLIR.StringRef.create(attr_str)
    attr = CAPI.mlirAttributeParseGet(ctx, attr)

    if is_null(attr) do
      raise "fail to parse attribute: #{attr_str}"
    end

    attr
  end

  def unit() do
    ctx = MLIR.Managed.Context.get()
    CAPI.mlirUnitAttrGet(ctx)
  end
end
