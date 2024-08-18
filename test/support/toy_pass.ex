defmodule ToyPass do
  use Beaver
  alias Beaver.MLIR
  alias MLIR.Type
  alias MLIR.Dialect.{Func, TOSA}
  require Func
  require TOSA
  require Type

  @moduledoc false
  use MLIR.Pass, on: "builtin.module"
  import Beaver.Pattern

  defpat replace_add_op() do
    a = value()
    b = value()
    res = type()
    {op, _t} = TOSA.add(a, b) >>> {:op, [res]}

    rewrite op do
      {r, _} = TOSA.sub(a, b) >>> {:op, [res]}
      replace(op, with: r)
    end
  end

  def run(%MLIR.Operation{} = operation) do
    with 1 <- Beaver.Walker.regions(operation) |> Enum.count(),
         {:ok, _} <-
           MLIR.Pattern.apply_(MLIR.Module.from_operation(operation), [replace_add_op(benefit: 2)]) do
      :ok
    end
  end
end
