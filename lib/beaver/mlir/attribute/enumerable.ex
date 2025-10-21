defimpl Enumerable, for: Beaver.MLIR.Attribute do
  alias Beaver.MLIR
  defstruct num_elements: nil, get_element: nil, index: 0

  def new(attr) do
    accessor = MLIR.Attribute.Accessor.new(attr)

    %__MODULE__{
      num_elements: accessor.get_num_element.(attr),
      get_element: &accessor.get_element.(attr, &1)
    }
  end

  def count(attr) do
    {:ok, new(attr).num_elements}
  end

  defguardp is_term_element(a, b)
            when (is_integer(a) and is_integer(b)) or (is_float(a) and is_float(b)) or
                   (is_boolean(a) and is_boolean(b)) or (is_binary(a) and is_binary(b))

  def member?(attr, element) do
    %__MODULE__{num_elements: num_elements, get_element: get_element} = new(attr)

    {:ok,
     Enum.any?(0..(num_elements - 1), fn pos ->
       case {get_element.(pos), element} do
         {%MLIR.NamedAttribute{} = a, %MLIR.NamedAttribute{} = b} ->
           MLIR.equal?(MLIR.NamedAttribute.name(a), MLIR.NamedAttribute.name(b)) and
             MLIR.equal?(MLIR.NamedAttribute.attribute(a), MLIR.NamedAttribute.attribute(b))

         {%MLIR.NamedAttribute{} = na, {name, attr}} ->
           MLIR.equal?(
             MLIR.NamedAttribute.name(na),
             MLIR.Identifier.get(name, ctx: MLIR.context(attr))
           ) and
             MLIR.equal?(MLIR.NamedAttribute.attribute(na), attr)

         {%m{} = a, %m{} = b} ->
           MLIR.equal?(a, b)

         {a, b} when is_term_element(a, b) ->
           a == b
       end
     end)}
  end

  def do_reduce(%__MODULE__{}, {:halt, acc}, _fun), do: {:halted, acc}
  def do_reduce(_attr, {:halt, acc}, _fun), do: {:halted, acc}

  def do_reduce(%__MODULE__{} = enum, {:suspend, acc}, fun),
    do: {:suspended, acc, &do_reduce(enum, &1, fun)}

  def do_reduce(attr, {:suspend, acc}, fun), do: {:suspended, acc, &do_reduce(attr, &1, fun)}

  def do_reduce(
        %__MODULE__{index: index, num_elements: num_elements, get_element: get_element} = enum,
        {:cont, acc},
        fun
      ) do
    if index < num_elements do
      head = get_element.(index)
      do_reduce(%__MODULE__{enum | index: index + 1}, fun.(head, acc), fun)
    else
      {:done, acc}
    end
  end

  def reduce(attr, {:cont, acc}, fun) do
    do_reduce(new(attr), {:cont, acc}, fun)
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
