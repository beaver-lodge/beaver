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
        {new_const, [%MLIR.Value{}]} =
          Arith.constant(value: Attribute.integer(MLIR.Type.i64(), 3)) >>> {:op, [Type.i64()]}

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

  test "with_default_config yields a usable greedy rewrite config" do
    assert :ok ==
             MLIR.Rewrite.with_default_config(fn cfg ->
               MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, 3)
               MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxNumRewrites(cfg, 7)

               assert 3 ==
                        cfg
                        |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxIterations()
                        |> Beaver.Native.to_term()

               assert 7 ==
                        cfg
                        |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxNumRewrites()
                        |> Beaver.Native.to_term()

               :ok
             end)
  end

  test "with_default_config accepts config opts" do
    assert {4, 9} ==
             MLIR.Rewrite.with_default_config(
               [max_iterations: 4, max_num_rewrites: 9],
               fn cfg ->
                 {
                   cfg
                   |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxIterations()
                   |> Beaver.Native.to_term(),
                   cfg
                   |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxNumRewrites()
                   |> Beaver.Native.to_term()
               }
             end
           )
  end

  test "with_default_config accepts config callbacks directly" do
    callback =
      fn cfg ->
        MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, 12)
        MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxNumRewrites(cfg, 21)
      end

    assert {12, 21} ==
             MLIR.Rewrite.with_default_config(callback, fn cfg ->
               {
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxIterations()
                 |> Beaver.Native.to_term(),
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxNumRewrites()
                 |> Beaver.Native.to_term()
               }
             end)
  end

  test "with_default_config accepts nil as the default config input" do
    assert {6, 13} ==
             MLIR.Rewrite.with_default_config(nil, fn cfg ->
               MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, 6)
               MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxNumRewrites(cfg, 13)

               {
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxIterations()
                 |> Beaver.Native.to_term(),
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxNumRewrites()
                 |> Beaver.Native.to_term()
               }
             end)
  end

  test "rewrite pattern apply accepts a config callback", %{ctx: ctx} do
    ir = example_ir(ctx)

    MLIR.Rewrite.apply_patterns!(
      ir,
      [
        {Arith.constant(), &constant_1_to_2/4},
        {Arith.constant(), &constant_2_to_3_with_op/4}
      ],
      fn cfg ->
        MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, 2)
        MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxNumRewrites(cfg, 8)

        assert 2 ==
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxIterations()
                 |> Beaver.Native.to_term()

        assert 8 ==
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxNumRewrites()
                 |> Beaver.Native.to_term()
      end
    )

    MLIR.verify!(ir)
  end

  test "rewrite pattern apply accepts config opts", %{ctx: ctx} do
    ir = example_ir(ctx)

    MLIR.Rewrite.apply_patterns!(
      ir,
      [
        {Arith.constant(), &constant_1_to_2/4},
        {Arith.constant(), &constant_2_to_3_with_op/4}
      ],
      max_iterations: 2,
      max_num_rewrites: 8
    )

    MLIR.verify!(ir)
  end

  test "rewrite pattern apply accepts nil config input", %{ctx: ctx} do
    ir = example_ir(ctx)

    MLIR.Rewrite.apply_patterns!(
      ir,
      [
        {Arith.constant(), &constant_1_to_2/4},
        {Arith.constant(), &constant_2_to_3_with_op/4}
      ],
      nil
    )

    MLIR.verify!(ir)
  end

  test "rewrite pattern apply returns diagnostics for list-based patterns", %{ctx: ctx} do
    ir = example_ir(ctx)

    {result, diagnostics} =
      MLIR.Rewrite.apply_patterns(
        ir,
        [
          {Arith.constant(), &constant_1_to_2/4},
          {Arith.constant(), &constant_2_to_3_with_op/4}
        ],
        nil
      )

    assert MLIR.LogicalResult.success?(result)
    assert diagnostics == []
    MLIR.verify!(ir)
  end

  test "rewrite pattern apply accepts mutable rewrite pattern sets directly", %{ctx: ctx} do
    ir = example_ir(ctx)
    set = MLIR.RewritePatternSet.create(ctx)

    MLIR.RewritePatternSet.add(set, Arith.constant(), &constant_1_to_2/4, ctx: ctx)
    MLIR.RewritePatternSet.add(set, Arith.constant(), &constant_2_to_3_with_op/4, ctx: ctx)

    {result, diagnostics} = MLIR.Rewrite.apply_patterns(ir, set, nil)

    assert MLIR.LogicalResult.success?(result)
    assert diagnostics == []
    MLIR.verify!(ir)
  end

  test "mlir apply accepts mutable rewrite pattern sets directly", %{ctx: ctx} do
    ir = example_ir(ctx)
    set = MLIR.RewritePatternSet.create(ctx)

    MLIR.RewritePatternSet.add(set, Arith.constant(), &constant_1_to_2/4, ctx: ctx)
    MLIR.RewritePatternSet.add(set, Arith.constant(), &constant_2_to_3_with_op/4, ctx: ctx)

    assert {:ok, ^ir} = MLIR.apply_(ir, set)
    MLIR.verify!(ir)
  end

  test "mlir apply forwards config callbacks for mutable rewrite pattern sets", %{ctx: ctx} do
    ir = example_ir(ctx)
    set = MLIR.RewritePatternSet.create(ctx)

    MLIR.RewritePatternSet.add(set, Arith.constant(), &constant_1_to_2/4, ctx: ctx)
    MLIR.RewritePatternSet.add(set, Arith.constant(), &constant_2_to_3_with_op/4, ctx: ctx)

    callback =
      fn cfg ->
        send(self(), :mutable_set_config_callback_called)
        MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, 2)
      end

    assert {:ok, ^ir} = MLIR.apply_(ir, set, callback)
    assert_received :mutable_set_config_callback_called
    MLIR.verify!(ir)
  end

  test "rewrite config opts can be converted into a reusable callback" do
    callback = MLIR.Rewrite.to_config_callback(max_iterations: 5, max_num_rewrites: 11)

    assert {5, 11} ==
             MLIR.Rewrite.with_default_config(fn cfg ->
               callback.(cfg)

               {
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxIterations()
                 |> Beaver.Native.to_term(),
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxNumRewrites()
                 |> Beaver.Native.to_term()
               }
             end)
  end

  test "rewrite config callbacks can be passed through unchanged" do
    callback =
      fn cfg ->
        MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, 6)
        MLIR.CAPI.mlirGreedyRewriteDriverConfigSetMaxNumRewrites(cfg, 13)
      end

    assert callback == MLIR.Rewrite.to_config_callback(callback)

    assert {6, 13} ==
             MLIR.Rewrite.with_default_config(fn cfg ->
               MLIR.Rewrite.to_config_callback(callback).(cfg)

               {
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxIterations()
                 |> Beaver.Native.to_term(),
                 cfg
                 |> MLIR.CAPI.mlirGreedyRewriteDriverConfigGetMaxNumRewrites()
                 |> Beaver.Native.to_term()
               }
             end)
  end
end
