defmodule Beaver.MLIR.Attribute do
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI
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

  def equal?(%MLIR.CAPI.MlirAttribute{} = a, %MLIR.CAPI.MlirAttribute{} = b) do
    CAPI.mlirAttributeEqual(a, b) |> Exotic.Value.extract()
  end

  def to_string(attr) do
    MLIR.StringRef.to_string(attr, CAPI, :mlirAttributePrint)
  end

  def float(type, value, opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirFloatAttrDoubleGet(ctx, type, value)
  end

  def dense_elements(elements, shaped_type) when is_list(elements) do
    num_elements = length(elements)
    elements = elements |> Exotic.Value.Array.from_list() |> Exotic.Value.get_ptr()
    CAPI.mlirDenseElementsAttrGet(shaped_type, num_elements, elements)
  end

  def array(elements) when is_list(elements) do
    array(elements, [])
  end

  def array(elements, opts) when is_list(elements) do
    num_elements = length(elements)
    elements = elements |> Exotic.Value.Array.from_list() |> Exotic.Value.get_ptr()
    CAPI.mlirArrayAttrGet(num_elements, elements, opts)
  end

  def string(str, opts \\ []) when is_binary(str) do
    CAPI.mlirStringAttrGet(MLIR.StringRef.create(str), opts)
  end
end
