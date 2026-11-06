defmodule ToyPass do
  @moduledoc false
  use Beaver
  alias MLIR.Dialect.TOSA
  use MLIR.Pass, on: "builtin.module"

  def replace_add_op(_pattern, op, rewriter, state) do
    [a, b] = Enum.to_list(Beaver.Walker.operands(op))
    type = op |> MLIR.Operation.result(0) |> MLIR.Value.type()

    mlir ctx: MLIR.context(op), ip: rewriter do
      result = TOSA.sub(a, b) >>> type
      MLIR.PatternRewriter.replace_op(rewriter, op, result)
    end

    {:ok, state}
  end

  def run(operation, state) do
    MLIR.Rewrite.apply_patterns!(operation, [{"tosa.add", &replace_add_op/4}])
    state
  end
end

defmodule ToyPassWithInit do
  @moduledoc false
  use Beaver
  use MLIR.Pass, on: "builtin.module"

  def initialize(ctx, nil) do
    frozen_set =
      MLIR.RewritePatternSet.create(ctx)
      |> MLIR.RewritePatternSet.add("tosa.add", &ToyPass.replace_add_op/4, ctx: ctx)
      |> MLIR.RewritePatternSet.freeze()

    {:ok, %{patterns: frozen_set, ctx: ctx, owning: true}}
  end

  def clone(state), do: %{state | owning: false}

  def destruct(%{patterns: patterns, ctx: ctx, owning: true}),
    do: MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, patterns)

  def destruct(_state), do: :ok

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
