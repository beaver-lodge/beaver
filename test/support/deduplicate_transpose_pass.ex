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

  def run(func, state) do
    func
    |> Beaver.Walker.prewalk(fn
      x ->
        with %MLIR.Operation{} <- x,
             "tosa.transpose" <- Beaver.MLIR.Operation.name(x),
             operands <- Beaver.Walker.operands(x),
             {:ok, transpose_input_op} <- MLIR.Value.owner(operands[0]),
             "tosa.transpose" <- Beaver.MLIR.Operation.name(transpose_input_op),
             {:ok, transpose_perm_attr} <- extract_perms(x),
             {:ok, transpose_input_perm_attr} <- extract_perms(transpose_input_op),
             true <- redundant?(transpose_perm_attr, transpose_input_perm_attr) do
          Beaver.Walker.replace(x, Beaver.Walker.operands(transpose_input_op)[0])
        else
          _ -> x
        end
    end)

    state
  end
end
