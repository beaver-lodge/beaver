defmodule Beaver.MLIR.Attribute.Accessor do
  @moduledoc false
  alias Beaver.MLIR
  import MLIR.CAPI
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

  defp repack_named_attribute_for_dict(key, update) do
    # if the key is string-like, create NamedAttribute so it can be used to create dictionary
    if not is_integer(key) and match?(%MLIR.Attribute{}, update) do
      MLIR.NamedAttribute.get(key, update)
    else
      update
    end
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

    with {:ok, key} <- normalize_key(key, num_elements),
         elements <-
           Range.new(0, num_elements - 1, 1)
           |> Enum.map(&get_element.(attr, &1)),
         index <- to_index(key, elements, ctx),
         false <- is_nil(index) do
      case fun.(Enum.at(elements, index)) do
        :pop ->
          __MODULE__.pop(acs, attr, key)

        {get, update} ->
          update = repack_named_attribute_for_dict(key, update)
          new_attr = getter.(List.replace_at(elements, index, update), ctx: ctx)
          {get, new_attr}
      end
    else
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

    with {:ok, key} <- normalize_key(key, num_elements),
         elements <-
           Range.new(0, num_elements - 1, 1)
           |> Enum.map(&get_element.(attr, &1)),
         index <- to_index(key, elements, ctx),
         false <- is_nil(index),
         true <- is_integer(index) do
      elements
      |> List.pop_at(index)
      |> then(fn {popped, remaining} ->
        {popped, getter.(remaining, ctx: ctx)}
      end)
    else
      _ -> {nil, attr}
    end
  end

  defp cast_null_attr(a) do
    if MLIR.null?(a) do
      :error
    else
      {:ok, a}
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
          %m{} = a when m in [Beaver.MLIR.Attribute, Beaver.MLIR.NamedAttribute] ->
            cast_null_attr(a)

          v when is_integer(v) or is_float(v) or is_binary(v) or is_boolean(v) ->
            {:ok, v}
        end

      _ ->
        :error
    end
  end

  defp wrap_get_num_element(%__MODULE__{} = acs) do
    update_in(acs.get_num_element, fn f -> &Beaver.Native.to_term(f.(&1)) end)
  end

  defp wrap_get_element(%__MODULE__{} = acs) do
    update_in(acs.get_element, fn f ->
      &case f.(&1, &2) do
        %MLIR.Attribute{} = a -> a
        %MLIR.NamedAttribute{} = na -> na
        %MLIR.StringRef{} = s -> s |> MLIR.to_string()
        native_val -> Beaver.Native.to_term(native_val)
      end
    end)
  end

  defp wrap_postprocessing(%__MODULE__{} = acs) do
    acs
    |> wrap_get_num_element
    |> wrap_get_element
  end

  defp dense_element_getter(element_type, attr, pos) do
    cond do
      MLIR.equal?(element_type, MLIR.Type.i1()) ->
        mlirDenseElementsAttrGetBoolValue(attr, pos)

      MLIR.Type.integer?(element_type) ->
        width = MLIR.Type.width(element_type)

        apply(
          MLIR.CAPI,
          if MLIR.Type.unsigned?(element_type) do
            :"mlirDenseElementsAttrGetUInt#{width}Value"
          else
            :"mlirDenseElementsAttrGetInt#{width}Value"
          end,
          [attr, pos]
        )

      MLIR.equal?(element_type, MLIR.Type.index()) ->
        mlirDenseElementsAttrGetIndexValue(attr, pos)

      MLIR.equal?(element_type, MLIR.Type.f32()) ->
        mlirDenseElementsAttrGetFloatValue(attr, pos)

      MLIR.equal?(element_type, MLIR.Type.f64()) ->
        mlirDenseElementsAttrGetDoubleValue(attr, pos)

      true ->
        mlirDenseElementsAttrGetStringValue(attr, pos)
    end
  end

  defp try_dense_array(attr) do
    cond do
      MLIR.Attribute.dense_bool_array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseBoolArrayGetElement/2,
          getter: &MLIR.Attribute.dense_array(&1, Beaver.Native.Bool, &2)
        }

      MLIR.Attribute.dense_i8_array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseI8ArrayGetElement/2,
          getter: &MLIR.Attribute.dense_array(&1, Beaver.Native.I8, &2)
        }

      MLIR.Attribute.dense_i16_array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseI16ArrayGetElement/2,
          getter: &MLIR.Attribute.dense_array(&1, Beaver.Native.I16, &2)
        }

      MLIR.Attribute.dense_i32_array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseI32ArrayGetElement/2,
          getter: &MLIR.Attribute.dense_array(&1, Beaver.Native.I32, &2)
        }

      MLIR.Attribute.dense_i64_array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseI64ArrayGetElement/2,
          getter: &MLIR.Attribute.dense_array(&1, Beaver.Native.I64, &2)
        }

      MLIR.Attribute.dense_f32_array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseF32ArrayGetElement/2,
          getter: &MLIR.Attribute.dense_array(&1, Beaver.Native.F32, &2)
        }

      MLIR.Attribute.dense_f64_array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &mlirDenseF64ArrayGetElement/2,
          getter: &MLIR.Attribute.dense_array(&1, Beaver.Native.F64, &2)
        }

      true ->
        nil
    end
  end

  def new(attr) do
    cond do
      MLIR.Attribute.array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirArrayAttrGetNumElements/1,
          get_element: &mlirArrayAttrGetElement/2,
          getter: &MLIR.Attribute.array/2
        }

      MLIR.Attribute.dictionary?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDictionaryAttrGetNumElements/1,
          get_element: fn
            attr, pos when is_integer(pos) ->
              mlirDictionaryAttrGetElement(attr, pos)

            attr, name when is_bitstring(name) or is_atom(name) ->
              mlirDictionaryAttrGetElementByName(attr, MLIR.StringRef.create(name))
          end,
          getter: &MLIR.Attribute.dictionary/2
        }

      da = try_dense_array(attr) ->
        da

      MLIR.Attribute.dense_elements?(attr) ->
        shaped_type = beaverDenseElementsAttrGetType(attr)
        element_type = MLIR.Type.element_type(shaped_type)

        %__MODULE__{
          get_num_element: &beaverShapedTypeGetNumElements(beaverDenseElementsAttrGetType(&1)),
          get_element: &dense_element_getter(element_type, &1, &2),
          getter: &MLIR.Attribute.dense_elements(&1, shaped_type, &2)
        }

      true ->
        raise ArgumentError, "not a container attribute"
    end
    |> wrap_postprocessing()
  end
end
