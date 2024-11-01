defmodule Beaver.MLIR.Attribute do
  @moduledoc """
  This module defines functions parsing and creating attributes in MLIR.
  """
  import Beaver.MLIR.CAPI

  alias Beaver.MLIR
  alias Beaver.MLIR.Type
  import Beaver.Sigils

  use Kinda.ResourceKind, forward_module: Beaver.Native

  def get(attr_str, opts \\ []) when is_binary(attr_str) do
    attr = MLIR.StringRef.create(attr_str)

    Beaver.Deferred.from_opts(opts, fn ctx ->
      attr = mlirAttributeParseGet(ctx, attr)

      if MLIR.null?(attr) do
        raise "fail to parse attribute: #{attr_str}"
      end

      attr
    end)
  end

  def dense_elements(elements, shaped_type \\ {:i, 8}, opts \\ [])

  def dense_elements(elements, {t, width}, opts)
      when is_binary(elements) and t in [:i, :f] and is_integer(width) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        et = apply(MLIR.Type, t, [width, [ctx: ctx]])
        str = MLIR.StringRef.create(elements)

        Type.ranked_tensor([byte_size(elements)], et)
        |> mlirDenseElementsAttrRawBufferGet(
          MLIR.StringRef.length(str),
          MLIR.StringRef.data(str) |> Beaver.Native.Array.as_opaque()
        )
      end
    )
  end

  def dense_elements(elements, shaped_type, opts) when is_list(elements) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        num_elements = length(elements)

        elements =
          elements
          |> Enum.map(&Beaver.Deferred.create(&1, ctx))
          |> Beaver.Native.array(MLIR.Attribute)

        mlirDenseElementsAttrGet(
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
        mlirArrayAttrGet(
          ctx,
          length(elements),
          elements
          |> Enum.map(&Beaver.Deferred.create(&1, ctx))
          |> Beaver.Native.array(MLIR.Attribute)
        )
      end
    )
  end

  def dense_array(elements, type, opts \\ []) when is_list(elements) do
    get =
      case type do
        Beaver.Native.I32 ->
          &mlirDenseI32ArrayGet/3

        Beaver.Native.I64 ->
          &mlirDenseI64ArrayGet/3
      end

    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        get.(
          ctx,
          length(elements),
          Beaver.Native.array(elements, type)
        )
      end
    )
  end

  def string(str, opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      &mlirStringAttrGet(&1, MLIR.StringRef.create(str))
    )
  end

  def type(t)
      when is_function(t, 1) do
    &type(t.(&1))
  end

  def type(%MLIR.Type{} = t) do
    mlirTypeAttrGet(t)
  end

  defp composite(%MLIR.Type{} = t, validate, get)
       when is_function(validate, 1) and is_function(get, 1) do
    if validate.(t), do: get.(t), else: raise(ArgumentError, "incompatible type #{to_string(t)}")
  end

  defp composite(t, validate, get) when is_function(t, 1), do: &composite(t.(&1), validate, get)

  def integer(t, value) when is_integer(value) do
    composite(
      t,
      &(MLIR.Type.integer?(&1) or MLIR.Type.index?(&1)),
      &mlirIntegerAttrGet(&1, value)
    )
  end

  def float(t, value) when is_float(value) do
    composite(t, &MLIR.Type.float?/1, &mlirFloatAttrDoubleGet(MLIR.context(&1), &1, value))
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

        mlirBoolAttrGet(ctx, value)
      end
    )
  end

  def affine_map(map) when is_function(map, 1) do
    &affine_map(map.(&1))
  end

  def affine_map(%MLIR.AffineMap{} = map) do
    mlirAffineMapAttrGet(map)
  end

  def unit(opts \\ []) do
    Beaver.Deferred.from_opts(
      opts,
      &mlirUnitAttrGet(&1)
    )
  end

  def flat_symbol_ref(symbol, opts \\ []) do
    symbol = MLIR.StringRef.create(symbol)

    Beaver.Deferred.from_opts(
      opts,
      &mlirFlatSymbolRefAttrGet(&1, symbol)
    )
  end

  def symbol_ref(symbol, nested_symbols \\ [], opts \\ []) do
    symbol = MLIR.StringRef.create(symbol)

    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        nested_arr =
          nested_symbols
          |> Enum.map(fn
            s when is_binary(s) -> flat_symbol_ref(s, ctx: ctx)
            %MLIR.Attribute{} = s -> s
          end)
          |> Beaver.Native.array(MLIR.Attribute)

        mlirSymbolRefAttrGet(ctx, symbol, length(nested_symbols), nested_arr)
      end
    )
  end

  def index(value, opts \\ []) when is_integer(value) do
    Beaver.Deferred.from_opts(opts, ~a{#{value} : index})
  end

  def null() do
    mlirAttributeGetNull()
  end

  defdelegate unwrap_type(type_attr), to: MLIR.CAPI, as: :mlirTypeAttrGetValue

  defdelegate unwrap_string(str_attr), to: MLIR.CAPI, as: :mlirStringAttrGetValue

  def unwrap(%__MODULE__{} = attribute) do
    cond do
      Beaver.Native.to_term(mlirAttributeIsAType(attribute)) ->
        unwrap_type(attribute)

      Beaver.Native.to_term(mlirAttributeIsAString(attribute)) ->
        unwrap_string(attribute)
    end
  end
end
