defmodule TraitTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR.Dialect.Func
  require Func
  @moduletag :smoke

  test "terminator?", %{ctx: ctx} do
    assert MLIR.Context.terminator?(ctx, "func.return")
    assert MLIR.Context.terminator?(ctx, "gpu.return")
    assert MLIR.Context.terminator?(ctx, "cf.br")

    m =
      mlir ctx: ctx do
        module do
          Func.return() >>> []
        end
      end

    refute MLIR.Operation.terminator?(MLIR.Operation.from_module(m))

    ops = MLIR.Module.body(m) |> Beaver.Walker.operations()
    assert MLIR.Operation.terminator?(ops[0])
  end
end
