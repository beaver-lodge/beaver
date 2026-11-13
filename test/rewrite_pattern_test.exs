defmodule RewritePatternTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR.Dialect.{Func, Arith}
  alias Beaver.MLIR.{CAPI, Rewrite, RewritePatternSet}
  alias Beaver.Native

  defmodule FailedToConvergeRewritePattern do
    use Beaver.MLIR.RewritePattern

    def construct(:init_state), do: [:init_state]

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
    function_module ctx, const(function_type: Type.function([], [Type.i64(), Type.i64()])) do
      block do
        v0 = Arith.constant(value: Attribute.integer(Type.i64(), 1)) >>> Type.i64()
        v1 = Arith.constant(value: Attribute.integer(Type.i64(), 3)) >>> Type.i64()
        Func.return(v0, v1) >>> []
      end
    end
    |> MLIR.verify!()
  end

  test "rewrite pattern set create and destroy", %{ctx: ctx} do
    assert set = %MLIR.RewritePatternSet{} = RewritePatternSet.create(ctx)
    assert pat = Beaver.MLIR.RewritePattern.create(Arith.constant(), ctx: ctx)
    assert set = RewritePatternSet.add(set, pat)
    assert :ok = RewritePatternSet.threaded_destroy(ctx, set)
  end

  test "rewrite pattern apply failed to converge", %{ctx: ctx} do
    ir = example_ir(ctx)
    assert set = RewritePatternSet.create(ctx)

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

    RewritePatternSet.add(set, pat)

    RewritePatternSet.add(set, Arith.constant(), FailedToConvergeRewritePattern, ctx: ctx, init_state: :init_state)

    assert frozen_set = %MLIR.FrozenRewritePatternSet{} = RewritePatternSet.freeze(set)

    assert_raise RuntimeError, "pattern application failed to converge", fn ->
      Rewrite.apply_patterns!(ir, frozen_set)
    end

    assert :ok = MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen_set)
  end

  def constant_1_to_2(_pattern, op, rewriter, state) do
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(op)
    one = MLIR.Attribute.integer(MLIR.Type.i64(ctx: ctx), 1)

    if MLIR.Operation.name(op) == Arith.constant() and MLIR.equal?(op[:value], one) do
      MLIR.PatternRewriter.with_insertion_point(rewriter, {:before, op}, fn ->
        replace_with_mapped_value(base, op, ctx)
      end)

      {:ok, state}
    else
      {:error, state}
    end
  end

  defp replace_with_mapped_value(base, op, ctx) do
    MLIR.IRMapping.with_mapping(fn mapping ->
      mlir ctx: ctx, ip: base do
        v = Arith.constant(value: Attribute.integer(MLIR.Type.i64(), 2)) >>> Type.i64()
        from = MLIR.Operation.result(op, 0)

        mapping
        |> MLIR.IRMapping.map(from, v)
        |> MLIR.IRMapping.lookup(from)
        |> then(&MLIR.RewriterBase.replace(base, from, &1))
      end
    end)
  end

  def constant_2_to_3_with_op(_pattern, op, rewriter, state) do
    base = MLIR.PatternRewriter.as_base(rewriter)
    ctx = MLIR.context(op)

    if MLIR.Operation.name(op) == Arith.constant() and
         MLIR.equal?(op[:value], MLIR.Attribute.integer(MLIR.Type.i64(ctx: ctx), 2)) do
      mlir ctx: ctx, ip: rewriter do
        {new_const, [%MLIR.Value{}]} =
          Arith.constant(value: Attribute.integer(MLIR.Type.i64(), 3)) >>> {:op, [Type.i64()]}

        MLIR.RewriterBase.replace_op(base, op, new_const)
      end

      {:ok, state}
    else
      {:error, state}
    end
  end

  test "rewrite pattern apply", %{ctx: ctx} do
    ir = example_ir(ctx)

    Rewrite.apply_patterns!(ir, patterns())

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

  defp config_values(cfg) do
    {CAPI.mlirGreedyRewriteDriverConfigGetMaxIterations(cfg) |> Native.to_term(),
     CAPI.mlirGreedyRewriteDriverConfigGetMaxNumRewrites(cfg) |> Native.to_term()}
  end

  defp set_config(cfg, {iterations, rewrites}) do
    CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, iterations)
    CAPI.mlirGreedyRewriteDriverConfigSetMaxNumRewrites(cfg, rewrites)
  end

  defp config_input(:opts, {iterations, rewrites}), do: [max_iterations: iterations, max_num_rewrites: rewrites]
  defp config_input(:callback, values), do: &set_config(&1, values)
  defp config_input(nil, _), do: nil
  defp patterns, do: [{Arith.constant(), &constant_1_to_2/4}, {Arith.constant(), &constant_2_to_3_with_op/4}]

  defp pattern_set(ctx) do
    Enum.reduce(patterns(), RewritePatternSet.create(ctx), fn {name, callback}, set ->
      RewritePatternSet.add(set, name, callback, ctx: ctx)
    end)
  end

  test "with_default_config yields a usable greedy rewrite config" do
    assert :ok ==
             Rewrite.with_default_config(fn cfg ->
               set_config(cfg, {3, 7})
               assert config_values(cfg) == {3, 7}
               :ok
             end)
  end

  for {kind, values} <- [opts: {4, 9}, callback: {12, 21}, nil: {6, 13}] do
    @tag config: kind, values: values
    test "with_default_config accepts #{kind}", %{config: kind, values: values} do
      assert values ==
               Rewrite.with_default_config(config_input(kind, values), fn cfg ->
                 if kind == nil, do: set_config(cfg, values)
                 config_values(cfg)
               end)
    end
  end

  for kind <- [:callback, :opts, nil] do
    @tag config: kind
    test "rewrite pattern apply accepts #{kind} config", %{ctx: ctx, config: kind} do
      ir = example_ir(ctx)

      config =
        if kind == :callback do
          fn cfg ->
            set_config(cfg, {2, 8})
            assert config_values(cfg) == {2, 8}
          end
        else
          config_input(kind, {2, 8})
        end

      Rewrite.apply_patterns!(ir, patterns(), config)
      MLIR.verify!(ir)
    end
  end

  test "rewrite pattern apply returns diagnostics for list-based patterns", %{ctx: ctx} do
    ir = example_ir(ctx)

    {result, diagnostics} = Rewrite.apply_patterns(ir, patterns(), nil)

    assert MLIR.LogicalResult.success?(result)
    assert diagnostics == []
    MLIR.verify!(ir)
  end

  test "rewrite pattern apply accepts mutable rewrite pattern sets directly", %{ctx: ctx} do
    ir = example_ir(ctx)
    set = pattern_set(ctx)
    {result, diagnostics} = Rewrite.apply_patterns(ir, set, nil)
    assert MLIR.LogicalResult.success?(result)
    assert diagnostics == []
    MLIR.verify!(ir)
  end

  test "mlir apply accepts mutable rewrite pattern sets directly", %{ctx: ctx} do
    ir = example_ir(ctx)
    set = pattern_set(ctx)
    assert {:ok, ^ir} = MLIR.apply_(ir, set)
    MLIR.verify!(ir)
  end

  test "mlir apply forwards config callbacks for mutable rewrite pattern sets", %{ctx: ctx} do
    ir = example_ir(ctx)
    set = pattern_set(ctx)

    callback = fn cfg ->
      send(self(), :mutable_set_config_callback_called)
      CAPI.mlirGreedyRewriteDriverConfigSetMaxIterations(cfg, 2)
    end

    assert {:ok, ^ir} = MLIR.apply_(ir, set, callback)
    assert_received :mutable_set_config_callback_called
    MLIR.verify!(ir)
  end

  for {kind, values} <- [opts: {5, 11}, callback: {6, 13}] do
    @tag config: kind, values: values
    test "rewrite config #{kind} produces a reusable callback", %{config: kind, values: values} do
      input = config_input(kind, values)
      callback = Rewrite.to_config_callback(input)
      if kind == :callback, do: assert(callback == input)

      assert values ==
               Rewrite.with_default_config(fn cfg ->
                 callback.(cfg)
                 config_values(cfg)
               end)
    end
  end
end
