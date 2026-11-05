alias Beaver.MLIR.Attribute

defimpl Collectable, for: Attribute do
  alias Beaver.MLIR

  def into(%Attribute{} = attr) do
    # dense elements will fail to create if init values are empty, so we lift the restriction here
    if not Enum.empty?(attr) and not Attribute.dense_elements?(attr),
      do: raise(ArgumentError, "cannot collect into an attribute that is not empty")

    collector_fun = fn
      acc, {:cont, elem} ->
        acc =
          case acc do
            # drop the existing elements
            %Attribute{} -> []
            _ -> Enum.to_list(acc)
          end

        if Attribute.dense_bool_array?(attr), do: [elem == true | acc], else: [elem | acc]

      acc, :done ->
        acc = Enum.reverse(acc)
        Attribute.Accessor.new(attr).getter.(acc, ctx: MLIR.context(attr))

      _acc, :halt ->
        :ok
    end

    {attr, collector_fun}
  end
end
