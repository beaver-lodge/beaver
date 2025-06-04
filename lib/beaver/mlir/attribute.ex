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
        element_type = mlirShapedTypeGetElementType(shaped_type)

        shaped_type_num_elements =
          beaverShapedTypeGetNumElements(shaped_type) |> Beaver.Native.to_term()

        if shaped_type_num_elements != num_elements and num_elements != 1 do
          raise ArgumentError,
                "number of elements #{num_elements} does not match shaped type #{shaped_type_num_elements}"
        end

        cond do
          Enum.all?(elements, &match?(%MLIR.StringRef{}, &1)) and length(elements) > 0 ->
            strings = Beaver.Native.array(elements, MLIR.StringRef, mut: true)
            mlirDenseElementsAttrStringGet(shaped_type, num_elements, strings)

          Enum.all?(elements, &is_binary/1) and length(elements) > 0 ->
            elements
            |> Enum.map(&MLIR.StringRef.create/1)
            |> dense_elements(shaped_type, opts)

          Enum.all?(elements, &match?(%MLIR.Attribute{}, &1)) and length(elements) > 0 ->
            attrs = Beaver.Native.array(elements, MLIR.Attribute)
            mlirDenseElementsAttrGet(shaped_type, num_elements, attrs)

          MLIR.Type.opaque?(element_type) ->
            strings =
              elements
              |> Enum.map(&MLIR.StringRef.create/1)
              |> Beaver.Native.array(MLIR.StringRef, mut: true)

            mlirDenseElementsAttrStringGet(shaped_type, num_elements, strings)

          MLIR.equal?(element_type, MLIR.Type.i(1, ctx: ctx)) ->
            elements = Enum.map(elements, &if(&1, do: 1, else: 0))

            mlirDenseElementsAttrBoolGet(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.CInt)
            )

          MLIR.equal?(element_type, MLIR.Type.i(8, ctx: ctx)) ->
            mlirDenseElementsAttrInt8Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.I8)
            )

          MLIR.equal?(element_type, MLIR.Type.i(8, ctx: ctx, signed: false)) ->
            mlirDenseElementsAttrUInt8Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.U8)
            )

          MLIR.equal?(element_type, MLIR.Type.i(16, ctx: ctx)) ->
            mlirDenseElementsAttrInt16Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.I16)
            )

          MLIR.equal?(element_type, MLIR.Type.i(16, ctx: ctx, signed: false)) ->
            mlirDenseElementsAttrUInt16Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.U16)
            )

          MLIR.equal?(element_type, MLIR.Type.i(32, ctx: ctx)) ->
            mlirDenseElementsAttrInt32Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.I32)
            )

          MLIR.equal?(element_type, MLIR.Type.i(32, ctx: ctx, signed: false)) ->
            mlirDenseElementsAttrUInt32Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.U32)
            )

          MLIR.equal?(element_type, MLIR.Type.i(64, ctx: ctx)) ->
            mlirDenseElementsAttrInt64Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.I64)
            )

          MLIR.equal?(element_type, MLIR.Type.i(64, ctx: ctx, signed: false)) ->
            mlirDenseElementsAttrUInt64Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.U64)
            )

          MLIR.equal?(element_type, MLIR.Type.index(ctx: ctx)) ->
            mlirDenseElementsAttrUInt64Get(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.U64)
            )

          MLIR.equal?(element_type, MLIR.Type.f(32, ctx: ctx)) ->
            mlirDenseElementsAttrFloatGet(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.F32)
            )

          MLIR.equal?(element_type, MLIR.Type.f(64, ctx: ctx)) ->
            mlirDenseElementsAttrDoubleGet(
              shaped_type,
              num_elements,
              Beaver.Native.array(elements, Beaver.Native.F64)
            )

          true ->
            raise ArgumentError,
                  "unsupported element type #{inspect(elements)}"
        end
      end
    )
  end

  def dense_elements(value, shaped_type, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        case value do
          true ->
            mlirDenseElementsAttrBoolSplatGet(Beaver.Deferred.create(shaped_type, ctx), true)

          false ->
            mlirDenseElementsAttrBoolSplatGet(Beaver.Deferred.create(shaped_type, ctx), false)

          v when is_integer(v) ->
            element_type = mlirShapedTypeGetElementType(Beaver.Deferred.create(shaped_type, ctx))

            cond do
              MLIR.equal?(element_type, MLIR.Type.i(8, ctx: ctx)) ->
                mlirDenseElementsAttrInt8SplatGet(Beaver.Deferred.create(shaped_type, ctx), v)

              MLIR.equal?(element_type, MLIR.Type.i(8, ctx: ctx, signed: false)) ->
                mlirDenseElementsAttrUInt8SplatGet(Beaver.Deferred.create(shaped_type, ctx), v)

              # there is no mlirDenseElementsAttrInt16SplatGet

              # there is no mlirDenseElementsAttrUInt16SplatGet

              MLIR.equal?(element_type, MLIR.Type.i(32, ctx: ctx)) ->
                mlirDenseElementsAttrInt32SplatGet(Beaver.Deferred.create(shaped_type, ctx), v)

              MLIR.equal?(element_type, MLIR.Type.i(32, ctx: ctx, signed: false)) ->
                mlirDenseElementsAttrUInt32SplatGet(Beaver.Deferred.create(shaped_type, ctx), v)

              MLIR.equal?(element_type, MLIR.Type.i(64, ctx: ctx)) ->
                mlirDenseElementsAttrInt64SplatGet(Beaver.Deferred.create(shaped_type, ctx), v)

              MLIR.equal?(element_type, MLIR.Type.i(64, ctx: ctx, signed: false)) ->
                mlirDenseElementsAttrUInt64SplatGet(Beaver.Deferred.create(shaped_type, ctx), v)

              true ->
                raise ArgumentError, "unsupported integer element type"
            end

          v when is_float(v) ->
            element_type = mlirShapedTypeGetElementType(Beaver.Deferred.create(shaped_type, ctx))

            cond do
              MLIR.equal?(element_type, MLIR.Type.f(32, ctx: ctx)) ->
                mlirDenseElementsAttrFloatSplatGet(Beaver.Deferred.create(shaped_type, ctx), v)

              MLIR.equal?(element_type, MLIR.Type.f(64, ctx: ctx)) ->
                mlirDenseElementsAttrDoubleSplatGet(Beaver.Deferred.create(shaped_type, ctx), v)

              true ->
                raise ArgumentError, "unsupported float element type"
            end

          _ ->
            raise ArgumentError, "unsupported splat value type"
        end
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
        Beaver.Native.Bool ->
          &mlirDenseBoolArrayGet/3

        Beaver.Native.I8 ->
          &mlirDenseI8ArrayGet/3

        Beaver.Native.I16 ->
          &mlirDenseI16ArrayGet/3

        Beaver.Native.I32 ->
          &mlirDenseI32ArrayGet/3

        Beaver.Native.I64 ->
          &mlirDenseI64ArrayGet/3

        Beaver.Native.F32 ->
          &mlirDenseF32ArrayGet/3

        Beaver.Native.F64 ->
          &mlirDenseF64ArrayGet/3
      end

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
        get.(
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
    Check if the attribute is #{type_name}
    """
    def unquote(:"#{helper_name}?")(%__MODULE__{} = t) do
      unquote(f)(t) |> Beaver.Native.to_term()
    end
  end

  defmodule Accessor do
    @moduledoc false
    defstruct [:get_num_element, :get_element, :getter]

    defp normalize_key(index, num_elements) when is_integer(index) do
      index =
        if index < 0 do
          num_elements + index
        else
          index
        end

      if index >= 0 and index < num_elements do
        {:ok, index}
      else
        :error
      end
    end

    defp normalize_key(name, _num_elements) do
      {:ok, name}
    end

    defp to_index(key, _, _) when is_integer(key) do
      key
    end

    defp to_index(key, elements, ctx) when is_binary(key) or is_atom(key) do
      Enum.find_index(elements, fn
        %MLIR.NamedAttribute{} = na ->
          MLIR.equal?(
            MLIR.NamedAttribute.name(na),
            MLIR.Identifier.get(key, ctx: ctx)
          )

        _ ->
          false
      end)
    end

    def get_and_update(
          %__MODULE__{
            get_num_element: get_num_element,
            get_element: get_element,
            getter: getter
          } = acs,
          attr,
          key,
          fun
        ) do
      num_elements = get_num_element.(attr)
      ctx = MLIR.context(attr)

      case normalize_key(key, num_elements) do
        {:ok, key} ->
          elements =
            Range.new(0, num_elements - 1, 1)
            |> Enum.map(&get_element.(attr, &1))

          index = to_index(key, elements, ctx)

          case fun.(Enum.at(elements, index)) do
            :pop ->
              __MODULE__.pop(acs, attr, key)

            {get, update} ->
              update =
                if is_integer(key) do
                  update
                else
                  case update do
                    %MLIR.NamedAttribute{} = na ->
                      na

                    %MLIR.Attribute{} = a ->
                      MLIR.NamedAttribute.get(key, a)
                  end
                end

              new_attr = getter.(List.replace_at(elements, index, update), ctx: ctx)
              {get, new_attr}
          end

        _ ->
          {nil, attr}
      end
    end

    def pop(
          %__MODULE__{
            get_num_element: get_num_element,
            get_element: get_element,
            getter: getter
          },
          attr,
          key
        ) do
      num_elements = get_num_element.(attr)

      ctx = MLIR.context(attr)

      case normalize_key(key, num_elements) do
        {:ok, key} ->
          elements =
            Range.new(0, num_elements - 1, 1)
            |> Enum.map(&get_element.(attr, &1))

          index = to_index(key, elements, ctx)

          elements
          |> List.pop_at(index)
          |> then(fn {popped, remaining} ->
            {popped, getter.(remaining, ctx: ctx)}
          end)

        _ ->
          {nil, attr}
      end
    end

    def fetch(
          %__MODULE__{
            get_num_element: get_num_element,
            get_element: get_element
          },
          attr,
          key
        ) do
      num_elements = get_num_element.(attr)

      case normalize_key(key, num_elements) do
        {:ok, key} ->
          case get_element.(attr, key) do
            %MLIR.Attribute{} = a ->
              if MLIR.null?(a) do
                :error
              else
                {:ok, a}
              end

            v ->
              {:ok, v}
          end

        _ ->
          :error
      end
    end
  end

  @doc false
  def accessor(attr) do
    cond do
      array?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirArrayAttrGetNumElements/1,
          get_element: &mlirArrayAttrGetElement/2,
          getter: &array/2
        }

      dictionary?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirDictionaryAttrGetNumElements/1,
          get_element: fn
            attr, pos when is_integer(pos) ->
              mlirDictionaryAttrGetElement(attr, pos)

            attr, name when is_bitstring(name) or is_atom(name) ->
              mlirDictionaryAttrGetElementByName(attr, MLIR.StringRef.create(name))
          end,
          getter: &dictionary/2
        }

      dense_bool_array?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseBoolArrayGetElement/2,
          getter: &dense_array(&1, Beaver.Native.Bool, &2)
        }

      dense_i8_array?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseI8ArrayGetElement/2,
          getter: &dense_array(&1, Beaver.Native.I8, &2)
        }

      dense_i16_array?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseI16ArrayGetElement/2,
          getter: &dense_array(&1, Beaver.Native.I16, &2)
        }

      dense_i32_array?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseI32ArrayGetElement/2,
          getter: &dense_array(&1, Beaver.Native.I32, &2)
        }

      dense_i64_array?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseI64ArrayGetElement/2,
          getter: &dense_array(&1, Beaver.Native.I64, &2)
        }

      dense_f32_array?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseF32ArrayGetElement/2,
          getter: &dense_array(&1, Beaver.Native.F32, &2)
        }

      dense_f64_array?(attr) ->
        %__MODULE__.Accessor{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseF64ArrayGetElement/2,
          getter: &dense_array(&1, Beaver.Native.F64, &2)
        }

      dense_elements?(attr) ->
        element_type = beaverDenseElementsAttrGetElementType(attr)
        shaped_type = beaverDenseElementsAttrGetType(attr)
        ctx = MLIR.context(attr)

        %__MODULE__.Accessor{
          get_num_element: &beaverDenseElementsGetNumElements/1,
          get_element: fn attr, pos ->
            cond do
              MLIR.equal?(element_type, MLIR.Type.i(1, ctx: ctx)) ->
                mlirDenseElementsAttrGetBoolValue(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.i(8, ctx: ctx)) ->
                mlirDenseElementsAttrGetInt8Value(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.i(16, ctx: ctx)) ->
                mlirDenseElementsAttrGetInt16Value(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.i(32, ctx: ctx)) ->
                mlirDenseElementsAttrGetInt32Value(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.i(64, ctx: ctx)) ->
                mlirDenseElementsAttrGetInt64Value(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.i(8, ctx: ctx, signed: false)) ->
                mlirDenseElementsAttrGetUInt8Value(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.i(16, ctx: ctx, signed: false)) ->
                mlirDenseElementsAttrGetUInt16Value(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.i(32, ctx: ctx, signed: false)) ->
                mlirDenseElementsAttrGetUInt32Value(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.i(64, ctx: ctx, signed: false)) ->
                mlirDenseElementsAttrGetUInt64Value(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.index(ctx: ctx)) ->
                mlirDenseElementsAttrGetIndexValue(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.f(32, ctx: ctx)) ->
                mlirDenseElementsAttrGetFloatValue(attr, pos)

              MLIR.equal?(element_type, MLIR.Type.f(64, ctx: ctx)) ->
                mlirDenseElementsAttrGetDoubleValue(attr, pos)

              true ->
                mlirDenseElementsAttrGetStringValue(attr, pos)
            end
          end,
          getter: &dense_elements(&1, shaped_type, &2)
        }

      true ->
        raise "not a container attribute"
    end
    |> then(fn %__MODULE__.Accessor{} = acs ->
      update_in(acs.get_num_element, fn f -> &Beaver.Native.to_term(f.(&1)) end)
    end)
    |> then(fn %__MODULE__.Accessor{} = acs ->
      update_in(acs.get_element, fn f ->
        fn attr, key ->
          case f.(attr, key) do
            %MLIR.Attribute{} = a -> a
            %MLIR.NamedAttribute{} = na -> na
            %MLIR.StringRef{} = s -> s |> MLIR.to_string()
            native_val -> Beaver.Native.to_term(native_val)
          end
        end
      end)
    end)
  end

  defguardp is_index_or_name(key) when is_integer(key) or is_binary(key) or is_atom(key)

  @impl Access
  def fetch(attr, key) when is_index_or_name(key) do
    accessor(attr) |> __MODULE__.Accessor.fetch(attr, key)
  end

  @impl Access
  def get_and_update(attr, key, fun) when is_index_or_name(key) do
    accessor(attr) |> __MODULE__.Accessor.get_and_update(attr, key, fun)
  end

  @impl Access
  def pop(attr, key) when is_index_or_name(key) do
    if MLIR.Attribute.dense_elements?(attr) do
      raise ArgumentError, "cannot pop from dense elements attribute"
    end

    accessor(attr) |> __MODULE__.Accessor.pop(attr, key)
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

  defimpl Collectable do
    defstruct getter: nil

    def into(%Beaver.MLIR.Attribute{} = attr) do
      # dense elements will fail to create if init values are empty, so we lift the restriction here
      if not Enum.empty?(attr) and not MLIR.Attribute.dense_elements?(attr) do
        raise ArgumentError, "cannot collect into an attribute that is not empty"
      end

      collector_fun = fn
        acc, {:cont, elem} ->
          acc =
            case acc do
              # drop the existing elements
              %Beaver.MLIR.Attribute{} -> []
              _ -> Enum.to_list(acc)
            end

          cond do
            MLIR.Attribute.dense_bool_array?(attr) ->
              [elem == true | acc]

            true ->
              [elem | acc]
          end

        acc, :done ->
          acc = Enum.reverse(acc)
          MLIR.Attribute.accessor(attr).getter.(acc, ctx: MLIR.context(attr))

        _acc, :halt ->
          :ok
      end

      {attr, collector_fun}
    end
  end

  defimpl Enumerable do
    alias Beaver.MLIR
    defstruct num_elements: nil, get_element: nil, index: 0

    def new(attr) do
      accessor = MLIR.Attribute.accessor(attr)

      %__MODULE__{
        num_elements: accessor.get_num_element.(attr),
        get_element: &accessor.get_element.(attr, &1)
      }
    end

    def count(attr) do
      {:ok, new(attr).num_elements}
    end

    def member?(attr, element) do
      %__MODULE__{num_elements: num_elements, get_element: get_element} = new(attr)

      {:ok,
       for pos <- 0..(num_elements - 1)//1, reduce: false do
         true ->
           true

         false ->
           case {get_element.(pos), element} do
             {%MLIR.NamedAttribute{} = a, %MLIR.NamedAttribute{} = b} ->
               MLIR.equal?(MLIR.NamedAttribute.name(a), MLIR.NamedAttribute.name(b)) and
                 MLIR.equal?(MLIR.NamedAttribute.attribute(a), MLIR.NamedAttribute.attribute(b))

             {%MLIR.Attribute{} = a, %MLIR.Attribute{} = b} ->
               MLIR.equal?(a, b)

             {%MLIR.Identifier{} = id, %MLIR.Identifier{} = id2} ->
               MLIR.equal?(id, id2)

             {%MLIR.NamedAttribute{} = na, {name, attr}} ->
               MLIR.equal?(
                 MLIR.NamedAttribute.name(na),
                 MLIR.Identifier.get(name, ctx: MLIR.context(attr))
               ) and
                 MLIR.equal?(MLIR.NamedAttribute.attribute(na), attr)

             {%m{} = a, %m{} = b} ->
               MLIR.equal?(a, b)

             {a, b} when is_integer(a) and is_integer(b) ->
               a == b

             {a, b} when is_float(a) and is_float(b) ->
               a == b

             {a, b} when is_boolean(a) and is_boolean(b) ->
               a == b

             {a, b} when is_binary(a) and is_binary(b) ->
               a == b
           end
       end}
    end

    def reduce(%__MODULE__{}, {:halt, acc}, _fun), do: {:halted, acc}
    def reduce(_attr, {:halt, acc}, _fun), do: {:halted, acc}

    def reduce(%__MODULE__{} = enum, {:suspend, acc}, fun),
      do: {:suspended, acc, &reduce(enum, &1, fun)}

    def reduce(attr, {:suspend, acc}, fun), do: {:suspended, acc, &reduce(attr, &1, fun)}

    def reduce(
          %__MODULE__{index: index, num_elements: num_elements, get_element: get_element} = enum,
          {:cont, acc},
          fun
        ) do
      if index < num_elements do
        head = get_element.(index)
        reduce(%__MODULE__{enum | index: index + 1}, fun.(head, acc), fun)
      else
        {:done, acc}
      end
    end

    def reduce(attr, {:cont, acc}, fun) do
      reduce(new(attr), {:cont, acc}, fun)
    end

    def slice(attr) do
      %__MODULE__{num_elements: num_elements, get_element: get_element} = new(attr)

      {:ok, num_elements,
       fn start, length, step ->
         for pos <- start..(start + length - 1)//step do
           get_element.(pos)
         end
       end}
    end
  end
end
