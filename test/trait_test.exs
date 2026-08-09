defmodule TraitTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR.Dialect.Func
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
    assert MLIR.Trait.has?(ops[0], :terminator)
  end

  test "attaches and queries built-in dynamic traits idempotently", %{ctx: ctx} do
    assert CompleteSlang
           |> then(&Beaver.Slang.load(ctx, &1))
           |> MLIR.LogicalResult.success?()

    assert MLIR.Trait.has?(ctx, "complete_slang.yield", :terminator)
    assert MLIR.Trait.has?(ctx, "complete_slang.scope", :isolated_from_above)
    assert MLIR.Trait.has?(ctx, "complete_slang.scope", :no_terminator)

    assert CompleteSlang
           |> then(&Beaver.Slang.load(ctx, &1))
           |> MLIR.LogicalResult.success?()

    assert :ok =
             MLIR.Trait.attach_all(
               ctx,
               CompleteSlang.__slang_dialect_name__(),
               CompleteSlang.__slang_traits__()
             )

    assert :ok = MLIR.Trait.attach(ctx, "complete_slang.yield", :terminator)
  end

  test "validates built-in trait declarations" do
    assert MLIR.Trait.normalize!([:terminator, :terminator]) == [:terminator]

    assert_raise ArgumentError, ~r/conflicting Slang traits/, fn ->
      MLIR.Trait.normalize!([:terminator, :no_terminator])
    end

    assert_raise ArgumentError, ~r/unsupported Slang traits/, fn ->
      MLIR.Trait.normalize!([:unknown])
    end
  end
end
