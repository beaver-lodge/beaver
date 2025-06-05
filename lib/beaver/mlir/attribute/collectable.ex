defimpl Collectable, for: Beaver.MLIR.Attribute do
  alias Beaver.MLIR

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

        if MLIR.Attribute.dense_bool_array?(attr) do
          [elem == true | acc]
        else
          [elem | acc]
        end

      acc, :done ->
        acc = Enum.reverse(acc)
        MLIR.Attribute.Accessor.new(attr).getter.(acc, ctx: MLIR.context(attr))

      _acc, :halt ->
        :ok
    end

    {attr, collector_fun}
  end
end
