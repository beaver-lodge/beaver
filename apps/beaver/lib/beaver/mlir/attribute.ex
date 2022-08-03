defmodule Beaver.MLIR.Attribute do
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def is_null(a) do
    CAPI.beaverAttributeIsNull(a) |> Beaver.Native.to_term()
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
    CAPI.mlirAttributeEqual(a, b) |> Beaver.Native.to_term()
  end

  def float(type, value, opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirFloatAttrDoubleGet(ctx, type, value)
  end

  def dense_elements(elements, shaped_type) when is_list(elements) do
    num_elements = length(elements)
    elements = elements |> CAPI.MlirAttribute.array()
    CAPI.mlirDenseElementsAttrGet(shaped_type, num_elements, elements)
  end

  def array(elements, opts \\ []) when is_list(elements) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    num_elements = length(elements)
    CAPI.mlirArrayAttrGet(ctx, num_elements, CAPI.MlirAttribute.array(elements))
  end

  def string(str, opts \\ []) when is_binary(str) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirStringAttrGet(ctx, MLIR.StringRef.create(str))
  end

  def type(%CAPI.MlirType{} = t) do
    CAPI.mlirTypeAttrGet(t)
  end

  def integer(%CAPI.MlirType{} = t, value) do
    CAPI.mlirIntegerAttrGet(t, value)
  end

  def bool(value, opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)

    value =
      case value do
        true -> 1
        false -> 0
      end

    CAPI.mlirBoolAttrGet(ctx, value)
  end
end
