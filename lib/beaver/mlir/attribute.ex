alias Beaver.MLIR.{Attribute, CAPI, NamedAttribute, StringRef, Type}
alias Beaver.{Deferred, Native}

defmodule Beaver.MLIR.Attribute do
  @moduledoc """
  This module defines functions parsing and creating attributes in MLIR.
  """
  import CAPI
  @behaviour Access

  alias Beaver.MLIR
  import Beaver.Sigils

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native

  defp raise_with_diagnostics(attr_str, diagnostics) do
    case diagnostics do
      [] ->
        raise ArgumentError, "fail to parse attribute: #{attr_str}"

      diagnostics when is_list(diagnostics) ->
        raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "fail to parse attribute")
    end
  end

  def get(attr_str, opts \\ []) when is_binary(attr_str) do
    attr = StringRef.create(attr_str)

    Deferred.from_opts(opts, fn ctx ->
      {attr, diagnostics} = mlirAttributeParseGetWithDiagnostics(ctx, ctx, attr)

      if MLIR.null?(attr), do: raise_with_diagnostics(attr_str, diagnostics), else: attr
    end)
  end

  defp truthy_elements(elements), do: Enum.map(elements, &if(&1, do: 1, else: 0))

  defp dense_elements_splat_get(shaped_type, value, ctx) do
    element_type = Type.element_type(shaped_type)

    cond do
      match?(%Attribute{}, value) ->
        mlirDenseElementsAttrSplatGet(shaped_type, value)

      is_boolean(value) ->
        mlirDenseElementsAttrBoolSplatGet(shaped_type, value)

      Type.integer?(element_type) ->
        width = Type.width(element_type)

        apply(
          CAPI,
          if(Type.unsigned?(element_type),
            do: :"mlirDenseElementsAttrUInt#{width}SplatGet",
            else: :"mlirDenseElementsAttrInt#{width}SplatGet"
          ),
          [shaped_type, value]
        )

      MLIR.equal?(element_type, Type.f(32, ctx: ctx)) ->
        mlirDenseElementsAttrFloatSplatGet(shaped_type, value)

      MLIR.equal?(element_type, Type.f(64, ctx: ctx)) ->
        mlirDenseElementsAttrDoubleSplatGet(shaped_type, value)

      true ->
        raise ArgumentError, "unsupported splat value type"
    end
  end

  defp string_elements?(elements) do
    (Enum.all?(elements, &is_binary/1) or Enum.all?(elements, &match?(%StringRef{}, &1))) and not Enum.empty?(elements)
  end

  defp attr_elements?(elements), do: Enum.all?(elements, &match?(%Attribute{}, &1)) and not Enum.empty?(elements)

  defp int_dense_elements_attr_getter(element_type) do
    width = Type.width(element_type)

    {f, t} =
      if Type.unsigned?(element_type),
        do: {:"mlirDenseElementsAttrUInt#{width}Get", :"Elixir.Beaver.Native.U#{width}"},
        else: {:"mlirDenseElementsAttrInt#{width}Get", :"Elixir.Beaver.Native.I#{width}"}

    &apply(CAPI, f, [&1, &2, Native.array(&3, t)])
  end

  defp float_dense_elements_attr_getter(element_type) do
    {f, t} =
      case Type.width(element_type) do
        32 -> {&mlirDenseElementsAttrFloatGet/3, Native.F32}
        64 -> {&mlirDenseElementsAttrDoubleGet/3, Native.F64}
      end

    &apply(f, [&1, &2, Native.array(&3, t)])
  end

  defp dense_elements_get(shaped_type, num_elements, elements) do
    element_type = Type.element_type(shaped_type)

    cond do
      Type.opaque?(element_type) or string_elements?(elements) ->
        strings =
          elements
          |> Enum.map(&StringRef.create/1)
          |> Native.array(StringRef, mut: true)

        mlirDenseElementsAttrStringGet(shaped_type, num_elements, strings)

      attr_elements?(elements) ->
        attrs = Native.array(elements, Attribute)
        mlirDenseElementsAttrGet(shaped_type, num_elements, attrs)

      MLIR.equal?(element_type, Type.i1()) ->
        elements = truthy_elements(elements)

        mlirDenseElementsAttrBoolGet(shaped_type, num_elements, Native.array(elements, Native.CInt))

      Type.integer?(element_type) ->
        int_dense_elements_attr_getter(element_type).(shaped_type, num_elements, elements)

      MLIR.equal?(element_type, Type.index()) ->
        mlirDenseElementsAttrUInt64Get(shaped_type, num_elements, Native.array(elements, Native.U64))

      Type.float?(element_type) ->
        float_dense_elements_attr_getter(element_type).(shaped_type, num_elements, elements)

      true ->
        raise ArgumentError,
              "unsupported element type #{MLIR.to_string(element_type)}"
    end
  end

  def dense_elements(elements_or_value, shaped_type \\ {:i, 8}, opts \\ [])

  def dense_elements(elements, {t, width}, opts)
      when is_binary(elements) and t in [:i, :f] and is_integer(width) do
    Deferred.from_opts(
      opts,
      fn ctx ->
        et = apply(Type, t, [width, [ctx: ctx]])
        str = StringRef.create(elements)

        Type.ranked_tensor!([byte_size(elements)], et)
        |> mlirDenseElementsAttrRawBufferGet(StringRef.length(str), StringRef.data(str) |> Native.Array.as_opaque())
      end
    )
  end

  def dense_elements(elements_or_value, shaped_type, opts) when is_list(elements_or_value) do
    Deferred.from_opts(
      opts,
      fn ctx ->
        shaped_type =
          case Deferred.resolve(shaped_type, ctx) do
            {:ok, %Type{} = t} -> t
            %Type{} = t -> t
          end

        elements = elements_or_value |> Enum.map(&Deferred.resolve(&1, ctx))
        num_elements = length(elements)

        shaped_type_num_elements = beaverShapedTypeGetNumElements(shaped_type) |> Native.to_term()

        if shaped_type_num_elements != num_elements and num_elements != 1 do
          raise ArgumentError,
                "number of elements #{num_elements} does not match shaped type #{shaped_type_num_elements}"
        end

        dense_elements_get(shaped_type, num_elements, elements)
      end
    )
  end

  def dense_elements(value, shaped_type, opts) do
    Deferred.from_opts(
      opts,
      fn ctx ->
        value = Deferred.resolve(value, ctx)
        shaped_type = Deferred.resolve(shaped_type, ctx)
        dense_elements_splat_get(shaped_type, value, ctx)
      end
    )
  end

  def array(elements, opts \\ []) when is_list(elements) do
    Deferred.from_opts(opts, fn ctx ->
      mlirArrayAttrGet(
        ctx,
        length(elements),
        elements |> Enum.map(&Deferred.resolve(&1, ctx)) |> Native.array(Attribute)
      )
    end)
  end

  defp dense_array_getter(Native.Bool), do: &mlirDenseBoolArrayGet/3
  defp dense_array_getter(Native.I8), do: &mlirDenseI8ArrayGet/3
  defp dense_array_getter(Native.I16), do: &mlirDenseI16ArrayGet/3
  defp dense_array_getter(Native.I32), do: &mlirDenseI32ArrayGet/3
  defp dense_array_getter(Native.I64), do: &mlirDenseI64ArrayGet/3
  defp dense_array_getter(Native.F32), do: &mlirDenseF32ArrayGet/3
  defp dense_array_getter(Native.F64), do: &mlirDenseF64ArrayGet/3

  def dense_array(elements, type, opts \\ []) when is_list(elements) do
    getter = dense_array_getter(type)

    {elements, type} =
      case type do
        Native.Bool -> {Enum.map(elements, &if(&1, do: 1, else: 0)), Native.CInt}
        _ -> {elements, type}
      end

    Deferred.from_opts(opts, fn ctx -> getter.(ctx, length(elements), Native.array(elements, type)) end)
  end

  def dictionary(elements, opts \\ []) when is_list(elements) do
    elements =
      elements
      |> Enum.map(fn
        %NamedAttribute{} = na -> na
        {name, attr} -> NamedAttribute.get(name, attr)
      end)

    Deferred.from_opts(opts, fn ctx ->
      mlirDictionaryAttrGet(
        ctx,
        length(elements),
        elements |> Enum.map(&Deferred.resolve(&1, ctx)) |> Native.array(NamedAttribute)
      )
    end)
  end

  def string(str, opts \\ []), do: Deferred.from_opts(opts, &mlirStringAttrGet(&1, StringRef.create(str)))

  def type(%Deferred{} = type), do: Deferred.defer(&type(Deferred.resolve(type, &1)))

  def type(%Type{} = t), do: mlirTypeAttrGet(t)

  defp composite(%Type{} = t, validate, get) when is_function(validate, 1) and is_function(get, 1),
    do: if(validate.(t), do: get.(t), else: raise(ArgumentError, "incompatible type #{to_string(t)}"))

  defp composite(%Deferred{} = type, validate, get),
    do: Deferred.defer(&composite(Deferred.resolve(type, &1), validate, get))

  def integer(t, value) when is_integer(value),
    do: composite(t, &(Type.integer?(&1) or Type.index?(&1)), &mlirIntegerAttrGet(&1, value))

  def float(t, value) when is_float(value),
    do: composite(t, &Type.float?/1, &mlirFloatAttrDoubleGet(MLIR.context(&1), &1, value))

  def bool(value, opts \\ []) do
    Deferred.from_opts(
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

  def affine_map(%Deferred{} = map), do: Deferred.defer(&affine_map(Deferred.resolve(map, &1)))

  def affine_map(%MLIR.AffineMap{} = map), do: mlirAffineMapAttrGet(map)

  def unit(opts \\ []), do: Deferred.from_opts(opts, &mlirUnitAttrGet(&1))

  def flat_symbol_ref(symbol, opts \\ []) do
    symbol = StringRef.create(symbol)

    Deferred.from_opts(opts, &mlirFlatSymbolRefAttrGet(&1, symbol))
  end

  def symbol_ref(symbol, nested_symbols \\ [], opts \\ []) do
    symbol = StringRef.create(symbol)

    Deferred.from_opts(
      opts,
      fn ctx ->
        nested_arr =
          nested_symbols
          |> Enum.map(fn
            s when is_binary(s) -> flat_symbol_ref(s, ctx: ctx)
            %Attribute{} = s -> s
          end)
          |> Native.array(Attribute)

        mlirSymbolRefAttrGet(ctx, symbol, length(nested_symbols), nested_arr)
      end
    )
  end

  def index(value, opts \\ []) when is_integer(value) do
    Deferred.from_opts(opts, fn ctx -> Deferred.resolve(~a{#{value} : index}, ctx) end)
  end

  def null(), do: mlirAttributeGetNull()

  for {f, "mlirAttributeIsA" <> type_name, 1} <-
        CAPI.__info__(:functions)
        |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end) do
    helper_name = type_name |> Macro.underscore()

    @doc """
    calls `Beaver.MLIR.CAPI.#{f}/1` to check if it is #{type_name} attribute
    """
    def unquote(:"#{helper_name}?")(%__MODULE__{} = attr),
      do: unquote(f)(attr) |> Native.to_term()
  end

  defguardp is_index_or_name(key) when is_integer(key) or is_binary(key) or is_atom(key)

  @impl Access
  def fetch(attr, key) when is_index_or_name(key),
    do: __MODULE__.Accessor.new(attr) |> __MODULE__.Accessor.fetch(attr, key)

  @impl Access
  def get_and_update(attr, key, fun) when is_index_or_name(key),
    do: __MODULE__.Accessor.new(attr) |> __MODULE__.Accessor.get_and_update(attr, key, fun)

  @impl Access
  def pop(attr, key) when is_index_or_name(key) do
    if Attribute.dense_elements?(attr),
      do: raise(ArgumentError, "cannot pop from dense elements attribute")

    __MODULE__.Accessor.new(attr) |> __MODULE__.Accessor.pop(attr, key)
  end

  @doc """
  Get the value of the attribute. The return type depends on the attribute type:
  """
  def value(%__MODULE__{} = attribute) do
    cond do
      type?(attribute) -> mlirTypeAttrGetValue(attribute)
      string?(attribute) -> mlirStringAttrGetValue(attribute) |> to_string()
      bool?(attribute) -> mlirBoolAttrGetValue(attribute) |> Native.to_term()
      integer?(attribute) -> mlirIntegerAttrGetValueInt(attribute) |> Native.to_term()
      float?(attribute) -> mlirFloatAttrGetValueDouble(attribute) |> Native.to_term()
      flat_symbol_ref?(attribute) -> mlirFlatSymbolRefAttrGetValue(attribute) |> to_string()
      unit?(attribute) -> :unit
    end
  end

  def strided_layout(offset, strides, opts \\ []) do
    Deferred.from_opts(opts, fn ctx ->
      mlirStridedLayoutAttrGet(
        ctx,
        MLIR.ShapedType.to_dynamic_magic_number(offset, :offset),
        length(strides),
        Native.array(Enum.map(strides, &MLIR.ShapedType.to_dynamic_magic_number(&1, :stride)), Native.I64)
      )
    end)
  end
end
