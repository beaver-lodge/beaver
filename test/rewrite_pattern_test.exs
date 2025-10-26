defmodule RewritePatternTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR.Dialect.{Func, Arith}
  require Func

  defmodule FailedToConvergeRewritePattern do
    use Beaver.MLIR.RewritePattern

    def construct(:init_state) do
      [:init_state]
    end

    def destruct(state) do
      [:init_state | tail] = Enum.reverse(state)
      Enum.all?(tail, &(:match_and_rewrite == &1)) || raise "State corrupted in destruct"
      :ok
    end

    def match_and_rewrite(_pattern, _op, rewriter, state) do
      %MLIR.RewriterBase{} = b = MLIR.PatternRewriter.as_base(rewriter)
      %MLIR.Context{} = MLIR.context(b)
      {:ok, [:match_and_rewrite | state]}
    end
  end

  defp example_ir(ctx) do
    mlir ctx: ctx do
      module do
        Func.func const(function_type: Type.function([], [Type.i64(), Type.i64()])) do
          region do
            block do
              v0 = Arith.constant(value: Attribute.integer(Type.i64(), 1)) >>> Type.i64()
              v1 = Arith.constant(value: Attribute.integer(Type.i64(), 3)) >>> Type.i64()
              Func.return(v0, v1) >>> []
            end
          end
        end
      end
    end
    |> MLIR.verify!()
  end

  test "rewrite pattern set create and destroy", %{ctx: ctx} do
    assert set = %MLIR.RewritePatternSet{} = MLIR.RewritePatternSet.create(ctx)
    assert pat = Beaver.MLIR.RewritePattern.create(Arith.constant(), ctx: ctx)
    assert set = MLIR.RewritePatternSet.add(set, pat)
    assert :ok = MLIR.RewritePatternSet.threaded_destroy(ctx, set)
  end

  test "rewrite pattern apply failed to converge", %{ctx: ctx} do
    ir = example_ir(ctx)
    assert set = MLIR.RewritePatternSet.create(ctx)

    assert pat =
             %MLIR.RewritePattern{} =
             MLIR.RewritePattern.create(Arith.constant(),
               ctx: ctx,
               benefit: 10,
               init_state: :init_state,
               construct: &FailedToConvergeRewritePattern.construct/1,
               destruct: &FailedToConvergeRewritePattern.destruct/1,
               match_and_rewrite: &FailedToConvergeRewritePattern.match_and_rewrite/4
             )

    MLIR.RewritePatternSet.add(set, pat)

    MLIR.RewritePatternSet.add(set, Arith.constant(), FailedToConvergeRewritePattern,
      ctx: ctx,
      init_state: :init_state
    )

    assert frozen_set = %MLIR.FrozenRewritePatternSet{} = MLIR.RewritePatternSet.freeze(set)

    assert_raise RuntimeError, "pattern application failed to converge", fn ->
      MLIR.Rewrite.apply_patterns!(ir, frozen_set)
    end

    assert :ok = MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen_set)
  end

  def constant_1_to_2(_pattern, op, rewriter, state) do
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(op)
    one = MLIR.Attribute.integer(MLIR.Type.i64(ctx: ctx), 1)

    if MLIR.Operation.name(op) == Arith.constant() and MLIR.equal?(op[:value], one) do
      mlir ctx: ctx, ip: base do
        v = Arith.constant(value: Attribute.integer(MLIR.Type.i64(), 2)) >>> Type.i64()
        MLIR.RewriterBase.replace(base, MLIR.Operation.result(op, 0), v)
      end

      {:ok, state}
    else
      {:error, state}
    end
  end

  def constant_2_to_3_with_op(_pattern, op, rewriter, state) do
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(op)

    if MLIR.Operation.name(op) == Arith.constant() and
         MLIR.equal?(op[:value], MLIR.Attribute.integer(MLIR.Type.i64(ctx: ctx), 2)) do
      mlir ctx: ctx, ip: rewriter do
        {new_const, _} =
          Arith.constant(value: Attribute.integer(MLIR.Type.i64(), 3)) >>> {:op, Type.i64()}

        MLIR.RewriterBase.replace(base, op, new_const)
      end

      {:ok, state}
    else
      {:error, state}
    end
  end

  test "rewrite pattern apply", %{ctx: ctx} do
    ir = example_ir(ctx)

    MLIR.Rewrite.apply_patterns!(ir, [
      {Arith.constant(), &constant_1_to_2/4},
      {Arith.constant(), &constant_2_to_3_with_op/4}
    ])

    ops =
      MLIR.Module.body(ir)
      |> Beaver.Walker.operations()
      |> Enum.at(0)
      |> MLIR.Dialect.Func.entry_block()
      |> Beaver.Walker.operations()

    # duplicated constants should be folded
    assert 2 = Enum.count(ops)
    assert Arith.constant() == MLIR.Operation.name(Enum.at(ops, 0))
    assert Func.return() == MLIR.Operation.name(Enum.at(ops, 1))
    MLIR.verify!(ir)
  end
end
