defmodule CallbackExecutionTest do
  use ExUnit.Case, async: true
  alias Beaver.{Composer, MLIR}
  alias Beaver.MLIR.Context

  defmodule Lifecycle do
    use MLIR.Pass, on: "func.func"
    def construct(state), do: tap(state, &send(self(), {:construct, &1}))
    def initialize(_ctx, nil), do: {:ok, :initialized}
    def clone(:initialized), do: 0

    def run(op, count) do
      send(self(), {:run, MLIR.Attribute.value(op[:sym_name]), count})
      count + 1
    end

    def destruct(state), do: send(self(), {:destruct, state})
  end

  defmodule Failing do
    use MLIR.Pass, on: "func.func"
    def initialize(_ctx, nil), do: {:ok, :initialized}
    def clone(:initialized), do: :cloned
    def run(_op, :cloned), do: raise("beam pass failed")
    def destruct(state), do: send(self(), {:failed_destruct, state})
  end

  defmodule InitFailure do
    use MLIR.Pass
    def initialize(_ctx, nil), do: {:error, :failed_init}
    def destruct(state), do: send(self(), {:init_destruct, state})
  end

  defmodule Filtered do
    use Beaver.Pattern.Native

    defrewrite wrong_arity(_op, _rewriter, _state), root: "arith.addi", operands: [_, _, _] do
      raise "native prefilter failed"
    end

    defrewrite missing_attribute(_op, _rewriter, _state),
      root: "arith.addi",
      attributes: [missing: _] do
      raise "native prefilter failed"
    end

    defrewrite missing_result(_op, _rewriter, _state), root: "arith.addi", results: [_, _] do
      raise "native prefilter failed"
    end
  end

  setup do
    ctx = Context.create(threading: false, thread_safe: true)
    on_exit(fn -> Context.destroy(ctx) end)
    %{ctx: ctx}
  end

  defp module(ctx, count \\ 2) do
    functions =
      for n <- 1..count do
        """
        func.func @f#{n}() -> i64 {
          %a = arith.constant 1 : i64
          %b = arith.constant 2 : i64
          %sum = arith.addi %a, %b : i64
          return %sum : i64
        }
        """
      end

    MLIR.Module.create!("module {#{Enum.join(functions)}}", ctx: ctx)
  end

  test "BEAM passes run in the caller with nested clone state and deterministic teardown", %{
    ctx: ctx
  } do
    ir = module(ctx)

    try do
      assert ir ==
               ir |> Composer.nested("func.func", Lifecycle) |> Composer.run!(execution: :beam)

      assert_received {:construct, nil}
      assert_received {:construct, 0}
      assert_received {:run, "f1", 0}
      assert_received {:run, "f2", 1}
      assert_received {:destruct, 2}
      assert_received {:destruct, :initialized}
      assert Context.thread_safe?(ctx)
      refute MLIR.CAPI.beaverContextIsMultithreadingEnabled(ctx) |> Beaver.Native.to_term()
    after
      MLIR.Module.destroy(ir)
    end
  end

  test "mixed native and BEAM stages retain per-operation pipeline order", %{ctx: ctx} do
    ir = module(ctx)

    observer = fn phase ->
      {Atom.to_string(phase), "func.func",
       fn op ->
         text = MLIR.to_string(op)

         send(
           self(),
           {phase, MLIR.Attribute.value(op[:sym_name]), String.contains?(text, "arith.addi")}
         )
       end}
    end

    try do
      Composer.nested(ir, "func.func", [observer.(:before), "canonicalize", observer.(:after)])
      |> Composer.run!(execution: :beam)

      assert_receive {:before, "f1", true}
      assert_receive {:after, "f1", false}
      assert_receive {:before, "f2", true}
      assert_receive {:after, "f2", false}
      assert MLIR.verify?(ir)
    after
      MLIR.Module.destroy(ir)
    end
  end

  test "BEAM failures stop later stages and clean both initialized and cloned state", %{ctx: ctx} do
    ir = module(ctx)

    try do
      assert {:error, diagnostics} =
               Composer.nested(ir, "func.func", [Failing, Lifecycle])
               |> Composer.run(execution: :beam)

      assert MLIR.Diagnostic.format(diagnostics) =~ "beam pass failed"
      assert_received {:failed_destruct, :cloned}
      assert_received {:failed_destruct, :initialized}
      assert_received {:destruct, 0}
      refute_received {:run, _, _}
      assert {:error, _} = Composer.append(ir, InitFailure) |> Composer.run(execution: :beam)
      assert_received {:init_destruct, :failed_init}
    after
      MLIR.Module.destroy(ir)
    end
  end

  test "independent locking supports native pass callbacks which apply nested patterns", %{
    ctx: ctx
  } do
    ir = module(ctx)
    caller = self()

    try do
      Composer.nested(
        ir,
        "func.func",
        {"nested-rewrite", "func.func",
         fn op ->
           MLIR.Rewrite.apply_patterns!(
             op,
             [
               {"arith.addi",
                fn _, _, _, state ->
                  send(caller, :pattern_called)
                  {:error, state}
                end}
             ],
             enable_folding: false
           )
         end}
      )
      |> Composer.run!()

      assert_receive :pattern_called
      assert_receive :pattern_called
    after
      MLIR.Module.destroy(ir)
    end
  end

  test "native prefilters reject before sending a BEAM match callback", %{ctx: ctx} do
    ir = module(ctx)
    set = MLIR.RewritePatternSet.create(ctx)

    patterns =
      for descriptor <- [
            Filtered.wrong_arity(),
            Filtered.missing_attribute(),
            Filtered.missing_result()
          ] do
        pattern =
          MLIR.RewritePattern.create(descriptor.root,
            ctx: ctx,
            prefilter: descriptor.prefilter,
            match_and_rewrite: descriptor.match_and_rewrite
          )

        MLIR.RewritePatternSet.add(set, pattern)
        pattern
      end

    frozen = MLIR.RewritePatternSet.freeze(set)

    try do
      MLIR.Rewrite.apply_patterns!(ir, frozen, enable_folding: false)
      for pattern <- patterns, do: assert(MLIR.RewritePattern.callback_count(pattern) == 0)
    after
      MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen)
      MLIR.Module.destroy(ir)
    end
  end

  test "a waiting callback does not starve a short compilation sharing its context", %{ctx: ctx} do
    long_ir = module(ctx, 1)
    short_ir = module(ctx, 1)
    parent = self()

    long =
      Task.async(fn ->
        Composer.append(
          long_ir,
          {"wait", "builtin.module",
           fn _ ->
             send(parent, {:waiting, self()})

             receive do
               :continue -> :ok
             end
           end}
        )
        |> Composer.run!()
      end)

    assert_receive {:waiting, callback}, 5_000

    try do
      short = Task.async(fn -> Composer.append(short_ir, "canonicalize") |> Composer.run!() end)
      assert short_ir == Task.await(short, 5_000)
    after
      send(callback, :continue)
      Task.await(long, 5_000)
      MLIR.Module.destroy(long_ir)
      MLIR.Module.destroy(short_ir)
    end
  end

  test "native filtering recognizes property-backed attributes", %{ctx: ctx} do
    ir = module(ctx, 1)
    set = MLIR.RewritePatternSet.create(ctx)

    pattern =
      MLIR.RewritePattern.create("arith.constant",
        ctx: ctx,
        prefilter: {0, 1, ["value"]},
        match_and_rewrite: fn _, _, _, state -> {:error, state} end
      )

    MLIR.RewritePatternSet.add(set, pattern)
    frozen = MLIR.RewritePatternSet.freeze(set)

    try do
      MLIR.Rewrite.apply_patterns!(ir, frozen, enable_folding: false)
      assert MLIR.RewritePattern.callback_count(pattern) >= 2
    after
      MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen)
      MLIR.Module.destroy(ir)
    end
  end

  test "pipeline verification detects invalid IR at the explicit stage boundary", %{ctx: ctx} do
    for verifier <- [true, :pipeline] do
      ir = module(ctx, 1)

      try do
        break_ir =
          {"change-signature", "func.func",
           fn op ->
             block = op |> MLIR.Dialect.Func.entry_block()

             MLIR.CAPI.mlirBlockAddArgument(
               block,
               MLIR.Type.i64(ctx: ctx),
               MLIR.Location.unknown(ctx: ctx)
             )
           end}

        observe = {"observe", "func.func", fn _ -> send(self(), {:after_invalid, verifier}) end}

        assert {:error, diagnostics} =
                 Composer.nested(ir, "func.func", [break_ir, observe])
                 |> Composer.run(execution: :beam, verifier: verifier)

        assert MLIR.Diagnostic.format(diagnostics) =~ "entry block"

        if verifier == :pipeline,
          do: assert_received({:after_invalid, :pipeline}),
          else: refute_received({:after_invalid, true})
      after
        MLIR.Module.destroy(ir)
      end
    end
  end

  test "allocation rewrites use PDL and agree across native and BEAM pass drivers", %{ctx: ctx} do
    source = """
    module {
      llvm.func @malloc(i64) -> !llvm.ptr
      llvm.func @free(!llvm.ptr)
      llvm.func @enif_alloc(i64) -> !llvm.ptr
      llvm.func @enif_free(!llvm.ptr)
      llvm.func @allocate(%size: i64) {
        %ptr = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
        llvm.call @free(%ptr) : (!llvm.ptr) -> ()
        llvm.return
      }
    }
    """

    first = MLIR.Module.create!(source, ctx: ctx)
    second = MLIR.Module.create!(source, ctx: ctx)

    try do
      for {ir, mode} <- [{first, :native}, {second, :beam}] do
        Composer.nested(ir, "llvm.func", Beaver.ENIF.UseENIFAlloc)
        |> Composer.run!(execution: mode)

        text = MLIR.to_string(ir)
        assert text =~ "llvm.call @enif_alloc"
        assert text =~ "llvm.call @enif_free"
        refute text =~ "llvm.call @malloc"
        refute text =~ "llvm.call @free"
      end

      assert MLIR.Operation.equivalent?(
               MLIR.Operation.from_module(first),
               MLIR.Operation.from_module(second),
               ignore_commutativity: true
             )
    after
      MLIR.Module.destroy(first)
      MLIR.Module.destroy(second)
    end
  end
end
