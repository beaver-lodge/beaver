defmodule TransposeHelper do
  @moduledoc false
  alias Beaver.MLIR.{Type, Attribute}
  def perm_t(), do: Type.ranked_tensor([2], Type.i32())
  def perms_t_attr(), do: Attribute.dense_array([1, 0], Beaver.Native.I32)
  def tensor_t(), do: Type.unranked_tensor(Type.f32())
end

defmodule DeduplicateTransposePass do
  @moduledoc false
  alias Beaver.MLIR
  use Beaver.MLIR.Pass, on: "func.func"
  alias Beaver.MLIR.Attribute

  def extract_perms(op) do
    if "tosa.transpose" == Beaver.MLIR.Operation.name(op) do
      {:ok, Beaver.Walker.attributes(op)["perms"]}
    end
  end

  def redundant?(%Attribute{} = attr1, %Attribute{} = attr2) do
    MLIR.equal?(attr1, attr2)
  end

  defp operations_in_order(operation) do
    {_, operations} =
      Beaver.Walker.postwalk(operation, [], fn
        %MLIR.Operation{} = op, acc -> {op, [op | acc]}
        element, acc -> {element, acc}
      end)

    Enum.reverse(operations)
  end

  defp remove_redundant_transpose(rewriter, operation) do
    with "tosa.transpose" <- MLIR.Operation.name(operation),
         operands <- Beaver.Walker.operands(operation),
         {:ok, transpose_input_op} <- MLIR.Value.owner(operands[0]),
         "tosa.transpose" <- MLIR.Operation.name(transpose_input_op),
         {:ok, transpose_perm_attr} <- extract_perms(operation),
         {:ok, transpose_input_perm_attr} <- extract_perms(transpose_input_op),
         true <- redundant?(transpose_perm_attr, transpose_input_perm_attr) do
      MLIR.RewriterBase.replace_op(
        rewriter,
        operation,
        Beaver.Walker.operands(transpose_input_op)[0]
      )
    else
      _ -> :ok
    end
  end

  def run(func, state) do
    operations = operations_in_order(func)

    MLIR.IRRewriter.with_rewriter(func, fn rewriter ->
      Enum.each(operations, &remove_redundant_transpose(rewriter, &1))
    end)

    state
  end
end
