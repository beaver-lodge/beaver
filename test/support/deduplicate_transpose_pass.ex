defmodule TransposeHelper do
  @moduledoc false
  alias Beaver.MLIR.{Type, Attribute}
  def perm_t(), do: Type.ranked_tensor([2], Type.i32())

  defp perm_int_attrs() do
    for perm <- 0..1, do: Attribute.integer(Type.i32(), perm)
  end

  def perms_t_attr(), do: Attribute.dense_elements(Enum.reverse(perm_int_attrs()), perm_t())
  def tensor_t(), do: Type.unranked_tensor(Type.f32())
end

defmodule DeduplicateTransposePass do
  @moduledoc false
  alias Beaver.MLIR
  use Beaver.MLIR.Pass, on: "func.func"
  alias Beaver.MLIR.Attribute

  def const_value(op) do
    operands = Beaver.Walker.operands(op)

    with true <- MLIR.Value.result?(operands[1]),
         const <- MLIR.CAPI.mlirOpResultGetOwner(operands[1]),
         "tosa.const" <- Beaver.MLIR.Operation.name(const) do
      {:ok, Beaver.Walker.attributes(const)["value"]}
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
             {:ok, transpose_perm_attr} <- const_value(x),
             {:ok, transpose_input_perm_attr} <- const_value(transpose_input_op),
             true <- redundant?(transpose_perm_attr, transpose_input_perm_attr) do
          Beaver.Walker.replace(x, Beaver.Walker.operands(transpose_input_op)[0])
        else
          _ -> x
        end
    end)

    state
  end
end
