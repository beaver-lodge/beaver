defmodule Beaver.MLIR.NamedAttribute do
  use Kinda.ResourceKind,
    forward_module: Beaver.Native
end

defmodule Beaver.MLIR.Attribute do
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI
  alias Beaver.MLIR

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  def is_null(a) do
    CAPI.beaverAttributeIsNull(a) |> Beaver.Native.to_term()
  end

  def get(attr_str, opts \\ []) when is_binary(attr_str) do
    attr = MLIR.StringRef.create(attr_str)

    Beaver.Deferred.from_opts(opts, fn ctx ->
      attr = CAPI.mlirAttributeParseGet(ctx, attr)

      if is_null(attr) do
        raise "fail to parse attribute: #{attr_str}"
      end

      attr
    end)
  end

  def equal?(%__MODULE__{} = a, %__MODULE__{} = b) do
    CAPI.mlirAttributeEqual(a, b) |> Beaver.Native.to_term()
  end

  def float(type, value, opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      &CAPI.mlirFloatAttrDoubleGet(&1, Beaver.Deferred.create(type, &1), value)
    )
  end

  def dense_elements(elements, shaped_type, opts \\ []) when is_list(elements) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        num_elements = length(elements)

        elements =
          elements
          |> Enum.map(&Beaver.Deferred.create(&1, ctx))
          |> Beaver.Native.array(MLIR.Attribute)

        CAPI.mlirDenseElementsAttrGet(
          Beaver.Deferred.create(shaped_type, ctx),
          num_elements,
          elements
        )
      end
    )
  end

  def array(elements, opts \\ []) when is_list(elements) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        CAPI.mlirArrayAttrGet(
          ctx,
          length(elements),
          elements
          |> Enum.map(&Beaver.Deferred.create(&1, ctx))
          |> Beaver.Native.array(MLIR.Attribute)
        )
      end
    )
  end

  def dense_array(elements, opts \\ []) when is_list(elements) do
    Beaver.Deferred.from_opts(
      opts,
      &CAPI.mlirDenseI32ArrayGet(
        &1,
        length(elements),
        Beaver.Native.array(elements, Beaver.Native.I32)
      )
    )
  end

  def string(str, opts \\ []) when is_binary(str) do
    Beaver.Deferred.from_opts(
      opts,
      &CAPI.mlirStringAttrGet(&1, MLIR.StringRef.create(str))
    )
  end

  def type(t)
      when is_function(t, 1) do
    &type(t.(&1))
  end

  def type(%MLIR.Type{} = t) do
    CAPI.mlirTypeAttrGet(t)
  end

  def integer(t, value) when is_function(t, 1) do
    fn ctx ->
      integer(t.(ctx), value)
    end
  end

  def integer(%MLIR.Type{} = t, value) do
    CAPI.mlirIntegerAttrGet(t, value)
  end

  def bool(value, opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        value =
          case value do
            true -> 1
            false -> 0
          end

        CAPI.mlirBoolAttrGet(ctx, value)
      end
    )
  end

  def affine_map(map) when is_function(map, 1) do
    &affine_map(map.(&1))
  end

  def affine_map(%MLIR.AffineMap{} = map) do
    MLIR.CAPI.mlirAffineMapAttrGet(map)
  end

  def unit(opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      &CAPI.mlirUnitAttrGet(&1)
    )
  end

  def flat_symbol_ref(symbol, opts \\ []) do
    symbol = MLIR.StringRef.create(symbol)

    Beaver.Deferred.from_opts(
      opts,
      &CAPI.mlirFlatSymbolRefAttrGet(&1, symbol)
    )
  end
end
