defmodule UndefinedDialect do
  # helper to generate a test dialect
  @moduledoc false
  alias Beaver.MLIR

  @dialect_name "testing"
  def foo(%Beaver.SSA{} = ssa) do
    MLIR.Operation.create(%Beaver.SSA{ssa | op: "#{@dialect_name}.foo"})
  end
end
