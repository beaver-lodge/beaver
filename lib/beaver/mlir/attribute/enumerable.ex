defimpl Enumerable, for: Beaver.MLIR.Attribute do
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
