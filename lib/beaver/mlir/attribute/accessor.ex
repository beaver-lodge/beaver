alias Beaver.MLIR.{Attribute, CAPI, NamedAttribute, Type}
alias Beaver.Native

defmodule Beaver.MLIR.Attribute.Accessor do
  @moduledoc false
  alias Beaver.MLIR
  import CAPI
  defstruct [:get_num_element, :get_element, :getter]

  defp normalize_key(index, num_elements) when is_integer(index) do
    index =
      if index < 0, do: num_elements + index, else: index

    if index >= 0 and index < num_elements, do: {:ok, index}, else: :error
  end

  defp normalize_key(name, _num_elements), do: {:ok, name}

  defp to_index(key, _, _) when is_integer(key), do: key

  defp to_index(key, elements, ctx) when is_binary(key) or is_atom(key) do
    Enum.find_index(elements, fn
      %NamedAttribute{} = na -> MLIR.equal?(NamedAttribute.name(na), MLIR.Identifier.get(key, ctx: ctx))
      _ -> false
    end)
  end

  defp repack_named_attribute_for_dict(key, update) do
    # if the key is string-like, create NamedAttribute so it can be used to create dictionary
    if not is_integer(key) and match?(%Attribute{}, update),
      do: NamedAttribute.get(key, update),
      else: update
  end

  def get_and_update(%__MODULE__{} = accessor, attr, key, fun) do
    count = accessor.get_num_element.(attr)
    ctx = MLIR.context(attr)

    with {:ok, key} <- normalize_key(key, count),
         elements <- Enum.map(0..(count - 1)//1, &accessor.get_element.(attr, &1)),
         index when is_integer(index) <- to_index(key, elements, ctx) do
      {get, elements} =
        case fun.(Enum.at(elements, index)) do
          :pop -> List.pop_at(elements, index)
          {get, update} -> {get, List.replace_at(elements, index, repack_named_attribute_for_dict(key, update))}
        end

      {get, accessor.getter.(elements, ctx: ctx)}
    else
      _ -> {nil, attr}
    end
  end

  def pop(accessor, attr, key), do: get_and_update(accessor, attr, key, fn _ -> :pop end)

  defp cast_null_attr(a), do: if(MLIR.null?(a), do: :error, else: {:ok, a})

  def fetch(%__MODULE__{get_num_element: get_num_element, get_element: get_element}, attr, key) do
    num_elements = get_num_element.(attr)

    case normalize_key(key, num_elements) do
      {:ok, key} ->
        case get_element.(attr, key) do
          %m{} = a when m in [Attribute, NamedAttribute] -> cast_null_attr(a)
          v when is_integer(v) or is_float(v) or is_binary(v) or is_boolean(v) -> {:ok, v}
          :dynamic -> {:ok, :dynamic}
        end

      _ ->
        :error
    end
  end

  defp wrap_postprocessing(%__MODULE__{get_num_element: count, get_element: get} = accessor) do
    %{
      accessor
      | get_num_element: &Native.to_term(count.(&1)),
        get_element: fn attr, key ->
          case get.(attr, key) do
            %kind{} = value when kind in [Attribute, NamedAttribute] -> value
            %MLIR.StringRef{} = string -> MLIR.to_string(string)
            :dynamic -> :dynamic
            value -> Native.to_term(value)
          end
        end
    }
  end

  defp dense_element_getter(element_type, attr, pos) do
    cond do
      MLIR.equal?(element_type, Type.i1()) ->
        mlirDenseElementsAttrGetBoolValue(attr, pos)

      Type.integer?(element_type) ->
        {suffix, _kind} = Attribute.integer_storage(element_type)
        apply(CAPI, :"mlirDenseElementsAttrGet#{suffix}Value", [attr, pos])

      MLIR.equal?(element_type, Type.index()) ->
        mlirDenseElementsAttrGetIndexValue(attr, pos)

      MLIR.equal?(element_type, Type.f32()) ->
        mlirDenseElementsAttrGetFloatValue(attr, pos)

      MLIR.equal?(element_type, Type.f64()) ->
        mlirDenseElementsAttrGetDoubleValue(attr, pos)

      true ->
        mlirDenseElementsAttrGetStringValue(attr, pos)
    end
  end

  @dense_array_types [
    {:bool, "Bool", Native.Bool},
    {:i8, "I8", Native.I8},
    {:i16, "I16", Native.I16},
    {:i32, "I32", Native.I32},
    {:i64, "I64", Native.I64},
    {:f32, "F32", Native.F32},
    {:f64, "F64", Native.F64}
  ]

  defp try_dense_array(attr) do
    Enum.find_value(@dense_array_types, fn {name, suffix, type} ->
      if apply(Attribute, :"dense_#{name}_array?", [attr]),
        do: %__MODULE__{
          get_num_element: &mlirDenseArrayGetNumElements/1,
          get_element: &apply(CAPI, :"mlirDense#{suffix}ArrayGetElement", [&1, &2]),
          getter: &Attribute.dense_array(&1, type, &2)
        }
    end)
  end

  def new(attr) do
    cond do
      Attribute.array?(attr) ->
        %__MODULE__{
          get_num_element: &mlirArrayAttrGetNumElements/1,
          get_element: &mlirArrayAttrGetElement/2,
          getter: &Attribute.array/2
        }

      Attribute.dictionary?(attr) ->
        %__MODULE__{
          get_num_element: &mlirDictionaryAttrGetNumElements/1,
          get_element: fn
            attr, pos when is_integer(pos) ->
              mlirDictionaryAttrGetElement(attr, pos)

            attr, name when is_bitstring(name) or is_atom(name) ->
              mlirDictionaryAttrGetElementByName(attr, MLIR.StringRef.create(name))
          end,
          getter: &Attribute.dictionary/2
        }

      da = try_dense_array(attr) ->
        da

      Attribute.dense_elements?(attr) ->
        shaped_type = beaverDenseElementsAttrGetType(attr)
        element_type = Type.element_type(shaped_type)

        %__MODULE__{
          get_num_element: &beaverShapedTypeGetNumElements(beaverDenseElementsAttrGetType(&1)),
          get_element: &dense_element_getter(element_type, &1, &2),
          getter: &Attribute.dense_elements(&1, shaped_type, &2)
        }

      Attribute.strided_layout?(attr) ->
        offset =
          mlirStridedLayoutAttrGetOffset(attr) |> MLIR.ShapedType.cast_dynamic_magic_number(:offset)

        %__MODULE__{
          get_num_element: &mlirStridedLayoutAttrGetNumStrides/1,
          get_element: fn
            _, :offset -> offset
            attr, pos -> mlirStridedLayoutAttrGetStride(attr, pos) |> MLIR.ShapedType.cast_dynamic_magic_number(:stride)
          end,
          getter: &Attribute.strided_layout(offset, &1, &2)
        }

      true ->
        raise ArgumentError, "not a container attribute"
    end
    |> wrap_postprocessing()
  end
end
