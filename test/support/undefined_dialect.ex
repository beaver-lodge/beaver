defmodule UndefinedDialect do
  # helper to generate a test dialect
  @moduledoc false
  alias Beaver.MLIR

  @dialect_name "testing"
  def foo(%Beaver.SSA{arguments: arguments, blk: block, ctx: ctx, results: results}) do
    MLIR.Operation.create_and_append(
      ctx,
      "#{@dialect_name}.foo",
      arguments,
      results,
      block
    )
  end
end
