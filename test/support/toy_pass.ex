defmodule ToyPass do
  @moduledoc false
  use Beaver
  alias MLIR.Dialect.{Func, TOSA}
  require Func
  import Beaver.Pattern
  use MLIR.Pass, on: "builtin.module"

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

  def run(%MLIR.Operation{} = operation, _state) do
    with 1 <- Beaver.Walker.regions(operation) |> Enum.count(),
         {:ok, _} <-
           MLIR.apply_(MLIR.Module.from_operation(operation), [replace_add_op(benefit: 2)]) do
      :ok
    else
      _ -> raise "unreachable"
    end
  end
end

defmodule ToyPassWithInit do
  @moduledoc false
  use Beaver
  use MLIR.Pass, on: "builtin.module"

  def initialize(ctx, nil) do
    frozen_pat_set = Beaver.Pattern.compile_patterns(ctx, [ToyPass.replace_add_op(benefit: 2)])
    {:ok, %{patterns: frozen_pat_set, owned: true}}
  end

  def clone(%{patterns: frozen_pat_set}) do
    %{patterns: frozen_pat_set, owned: false}
  end

  def destruct(%{patterns: frozen_pat_set, owned: true}) do
    MLIR.CAPI.mlirFrozenRewritePatternSetDestroy(frozen_pat_set)
  end

  def run(%MLIR.Operation{} = operation, %{patterns: patterns} = state) do
    with 1 <- Beaver.Walker.regions(operation) |> Enum.count(),
         {:ok, _} <-
           MLIR.apply_(MLIR.Module.from_operation(operation), patterns) do
      :ok
    else
      _ -> raise "unreachable"
    end

    state
  end
end
