defmodule Beaver.MLIR.Attribute do
  @moduledoc """
  This module defines functions parsing and creating attributes in MLIR.
  """
  import Beaver.MLIR.CAPI
  @behaviour Access

  alias Beaver.MLIR
  import Beaver.Sigils

  use Kinda.ResourceKind, forward_module: Beaver.Native

  defp raise_with_diagnostics(attr_str, diagnostics) do
    case diagnostics do
      [] ->
        raise ArgumentError, "fail to parse attribute: #{attr_str}"

      diagnostics when is_list(diagnostics) ->
        raise ArgumentError, MLIR.Diagnostic.format(diagnostics, "fail to parse attribute")
    end
  end

  def get(attr_str, opts \\ []) when is_binary(attr_str) do
    attr = MLIR.StringRef.create(attr_str)

    Beaver.Deferred.from_opts(opts, fn ctx ->
      {attr, diagnostics} = mlirAttributeParseGetWithDiagnostics(ctx, ctx, attr)

      if MLIR.null?(attr) do
        raise_with_diagnostics(attr_str, diagnostics)
      else
        attr
      end
    end)
  end

  defp truthy_elements(elements) do
    Enum.map(elements, &if(&1, do: 1, else: 0))
  end

  defp dense_elements_splat_get(shaped_type, value, ctx) do
    element_type = MLIR.Type.element_type(shaped_type)

    cond do
      match?(%MLIR.Attribute{}, value) ->
        mlirDenseElementsAttrSplatGet(shaped_type, value)

      is_boolean(value) ->
        mlirDenseElementsAttrBoolSplatGet(shaped_type, value)

      MLIR.Type.integer?(element_type) ->
        width = MLIR.Type.width(element_type)

        apply(
          MLIR.CAPI,
          if MLIR.Type.unsigned?(element_type) do
            :"mlirDenseElementsAttrUInt#{width}SplatGet"
          else
            :"mlirDenseElementsAttrInt#{width}SplatGet"
          end,
          [shaped_type, value]
        )

      MLIR.equal?(element_type, MLIR.Type.f(32, ctx: ctx)) ->
        mlirDenseElementsAttrFloatSplatGet(shaped_type, value)

      MLIR.equal?(element_type, MLIR.Type.f(64, ctx: ctx)) ->
        mlirDenseElementsAttrDoubleSplatGet(shaped_type, value)

      true ->
        raise ArgumentError, "unsupported splat value type"
    end
  end

  defp string_elements?(elements) do
    (Enum.all?(elements, &is_binary/1) or
       Enum.all?(elements, &match?(%MLIR.StringRef{}, &1))) and
      length(elements) > 0
  end

  defp attr_elements?(elements) do
    Enum.all?(elements, &match?(%MLIR.Attribute{}, &1)) and length(elements) > 0
  end

  defp int_dense_elements_attr_getter(element_type) do
    width = MLIR.Type.width(element_type)

    {f, t} =
      if MLIR.Type.unsigned?(element_type) do
        {:"mlirDenseElementsAttrUInt#{width}Get", :"Elixir.Beaver.Native.U#{width}"}
      else
        {:"mlirDenseElementsAttrInt#{width}Get", :"Elixir.Beaver.Native.I#{width}"}
      end

    &apply(MLIR.CAPI, f, [&1, &2, Beaver.Native.array(&3, t)])
  end

  defp float_dense_elements_attr_getter(element_type) do
    {f, t} =
      case MLIR.Type.width(element_type) do
        32 ->
          {&mlirDenseElementsAttrFloatGet/3, Beaver.Native.F32}

        64 ->
          {&mlirDenseElementsAttrDoubleGet/3, Beaver.Native.F64}
      end

    &apply(f, [&1, &2, Beaver.Native.array(&3, t)])
  end

  defp dense_elements_get(shaped_type, num_elements, elements) do
    element_type = MLIR.Type.element_type(shaped_type)

    cond do
      MLIR.Type.opaque?(element_type) or string_elements?(elements) ->
        strings =
          elements
          |> Enum.map(&MLIR.StringRef.create/1)
          |> Beaver.Native.array(MLIR.StringRef, mut: true)

        mlirDenseElementsAttrStringGet(shaped_type, num_elements, strings)

      attr_elements?(elements) ->
        attrs = Beaver.Native.array(elements, MLIR.Attribute)
        mlirDenseElementsAttrGet(shaped_type, num_elements, attrs)

      MLIR.equal?(element_type, MLIR.Type.i1()) ->
        elements = truthy_elements(elements)

        mlirDenseElementsAttrBoolGet(
          shaped_type,
          num_elements,
          Beaver.Native.array(elements, Beaver.Native.CInt)
        )

      MLIR.Type.integer?(element_type) ->
        int_dense_elements_attr_getter(element_type).(shaped_type, num_elements, elements)

      MLIR.equal?(element_type, MLIR.Type.index()) ->
        mlirDenseElementsAttrUInt64Get(
          shaped_type,
          num_elements,
          Beaver.Native.array(elements, Beaver.Native.U64)
        )

      MLIR.Type.float?(element_type) ->
        float_dense_elements_attr_getter(element_type).(shaped_type, num_elements, elements)

      true ->
        raise ArgumentError,
              "unsupported element type #{MLIR.to_string(element_type)}"
    end
  end

  def dense_elements(elements_or_value, shaped_type \\ {:i, 8}, opts \\ [])

  def dense_elements(elements, {t, width}, opts)
      when is_binary(elements) and t in [:i, :f] and is_integer(width) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        et = apply(MLIR.Type, t, [width, [ctx: ctx]])
        str = MLIR.StringRef.create(elements)

        MLIR.Type.ranked_tensor!([byte_size(elements)], et)
        |> mlirDenseElementsAttrRawBufferGet(
          MLIR.StringRef.length(str),
          MLIR.StringRef.data(str) |> Beaver.Native.Array.as_opaque()
        )
      end
    )
  end

  def dense_elements(elements_or_value, shaped_type, opts) when is_list(elements_or_value) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        shaped_type =
          case Beaver.Deferred.create(shaped_type, ctx) do
            {:ok, %Beaver.MLIR.Type{} = t} -> t
            %Beaver.MLIR.Type{} = t -> t
          end

        elements = elements_or_value |> Enum.map(&Beaver.Deferred.create(&1, ctx))
        num_elements = length(elements)

        shaped_type_num_elements =
          beaverShapedTypeGetNumElements(shaped_type) |> Beaver.Native.to_term()

        if shaped_type_num_elements != num_elements and num_elements != 1 do
          raise ArgumentError,
                "number of elements #{num_elements} does not match shaped type #{shaped_type_num_elements}"
        end

        dense_elements_get(shaped_type, num_elements, elements)
      end
    )
  end

  def dense_elements(value, shaped_type, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        value = Beaver.Deferred.create(value, ctx)
        shaped_type = Beaver.Deferred.create(shaped_type, ctx)
        dense_elements_splat_get(shaped_type, value, ctx)
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

  defp dense_array_getter(Beaver.Native.Bool), do: &mlirDenseBoolArrayGet/3
  defp dense_array_getter(Beaver.Native.I8), do: &mlirDenseI8ArrayGet/3
  defp dense_array_getter(Beaver.Native.I16), do: &mlirDenseI16ArrayGet/3
  defp dense_array_getter(Beaver.Native.I32), do: &mlirDenseI32ArrayGet/3
  defp dense_array_getter(Beaver.Native.I64), do: &mlirDenseI64ArrayGet/3
  defp dense_array_getter(Beaver.Native.F32), do: &mlirDenseF32ArrayGet/3
  defp dense_array_getter(Beaver.Native.F64), do: &mlirDenseF64ArrayGet/3

  def dense_array(elements, type, opts \\ []) when is_list(elements) do
    getter = dense_array_getter(type)

    {elements, type} =
      case type do
        Beaver.Native.Bool ->
          {Enum.map(elements, &if(&1, do: 1, else: 0)), Beaver.Native.CInt}

        _ ->
          {elements, type}
      end

    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        getter.(
          ctx,
          length(elements),
          Beaver.Native.array(elements, type)
        )
      end
    )
  end

  def dictionary(elements, opts \\ []) when is_list(elements) do
    elements =
      elements
      |> Enum.map(fn
        %MLIR.NamedAttribute{} = na -> na
        {name, attr} -> MLIR.NamedAttribute.get(name, attr)
      end)

    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        mlirDictionaryAttrGet(
          ctx,
          length(elements),
          elements
          |> Enum.map(&Beaver.Deferred.create(&1, ctx))
          |> Beaver.Native.array(MLIR.NamedAttribute)
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

  for {f, "mlirAttributeIsA" <> type_name, 1} <-
        Beaver.MLIR.CAPI.__info__(:functions)
        |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end) do
    helper_name = type_name |> Macro.underscore()

    @doc """
    calls `Beaver.MLIR.CAPI.#{f}/1` to check if it is #{type_name} attribute
    """
    def unquote(:"#{helper_name}?")(%__MODULE__{} = t) do
      unquote(f)(t) |> Beaver.Native.to_term()
    end
  end

  defguardp is_index_or_name(key) when is_integer(key) or is_binary(key) or is_atom(key)

  @impl Access
  def fetch(attr, key) when is_index_or_name(key) do
    __MODULE__.Accessor.new(attr) |> __MODULE__.Accessor.fetch(attr, key)
  end

  @impl Access
  def get_and_update(attr, key, fun) when is_index_or_name(key) do
    __MODULE__.Accessor.new(attr) |> __MODULE__.Accessor.get_and_update(attr, key, fun)
  end

  @impl Access
  def pop(attr, key) when is_index_or_name(key) do
    if MLIR.Attribute.dense_elements?(attr) do
      raise ArgumentError, "cannot pop from dense elements attribute"
    end

    __MODULE__.Accessor.new(attr) |> __MODULE__.Accessor.pop(attr, key)
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
