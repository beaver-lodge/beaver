defmodule Beaver.MLIR.Attribute.Accessor do
  @moduledoc false
  alias Beaver.MLIR
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
