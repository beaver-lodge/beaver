defmodule ExecutionOptimizationTest do
  use ExUnit.Case, async: true
  alias Beaver.{Composer, MLIR, Native, Walker}
  alias Beaver.Execution.Profile
  alias Beaver.MLIR.{CAPI, Context, IRBatch, Operation, Rewrite}
  alias Beaver.Pattern.Native.Program

  defmodule Lifecycle do
    use MLIR.Pass, on: "func.func"
    def construct(state), do: tap(state, &send(self(), {:construct, &1}))
    def initialize(_, nil), do: {:ok, :initialized}
    def clone(:initialized), do: 0

    def run(op, count) do
      send(self(), {:run, MLIR.Attribute.value(op[:sym_name]), count})
      count + 1
    end

    def destruct(state), do: send(self(), {:destruct, state})
  end

  defmodule Failing do
    use MLIR.Pass, on: "func.func"
    def initialize(_, nil), do: {:ok, :initialized}
    def clone(:initialized), do: :cloned
    def run(_, :cloned), do: raise("beam pass failed")
    def destruct(state), do: send(self(), {:failed_destruct, state})
  end

  defmodule InitFailure do
    use MLIR.Pass
    def initialize(_, nil), do: {:error, :failed_init}
    def destruct(state), do: send(self(), {:init_destruct, state})
  end

  defmodule Filtered do
    use Beaver.Pattern.Native

    defrewrite wrong_arity(_, _, _), root: "arith.addi", operands: [_, _, _] do
      raise "native prefilter failed"
    end

    defrewrite missing_attribute(_, _, _), root: "arith.addi", attributes: [missing: _] do
      raise "native prefilter failed"
    end

    defrewrite missing_result(_, _, _), root: "arith.addi", results: [_, _] do
      raise "native prefilter failed"
    end
  end

  setup do
    %{ctx: context()}
  end

  defp context(opts \\ []) do
    defaults = [
      threading: not Context.independent_locking_supported?(),
      thread_safe: true,
      callback_executor: [max_concurrency: 1, idle_timeout_ms: 25]
    ]

    ctx = Context.create(Keyword.merge(defaults, opts))
    on_exit(fn -> Context.destroy(ctx) end)
    ctx
  end

  @simple "module { func.func @f(%x: i64) -> i64 { %v = arith.ori %x, %x : i64 return %v : i64 } }"
  defp ir(ctx, source \\ @simple) do
    module = MLIR.Module.create!(source, ctx: ctx)
    on_exit(fn -> MLIR.Module.destroy(module) end)
    module
  end

  defp functions(ctx, count \\ 2) do
    body =
      Enum.map_join(
        1..count,
        &"func.func @f#{&1}() -> i64 { %a = arith.constant 1 : i64 %b = arith.constant 2 : i64 %sum = arith.addi %a, %b : i64 return %sum : i64 }"
      )

    ir(ctx, "module {#{body}}")
  end

  defp function(ir), do: ir |> MLIR.Module.body() |> Walker.operations() |> Enum.at(0)

  defp operations(ir),
    do:
      ir |> function() |> MLIR.Dialect.Func.entry_block() |> Walker.operations() |> Enum.to_list()

  defp concurrently(inputs, fun) do
    inputs |> Enum.map(&Task.async(fn -> fun.(&1) end)) |> Enum.each(&Task.await(&1, 5000))
  end

  defp nested(ir, pass), do: Composer.nested(ir, "func.func", pass)

  defp attr(op, name, value),
    do: CAPI.mlirOperationSetAttributeByName(op, MLIR.StringRef.create(name), value)

  defp break_signature(ir),
    do:
      CAPI.mlirBlockAddArgument(
        MLIR.Dialect.Func.entry_block(ir),
        MLIR.Type.i64(ctx: MLIR.context(ir)),
        MLIR.Location.unknown(ctx: MLIR.context(ir))
      )

  test "verifier strategies preserve diagnostics and record bounded per-run costs", %{ctx: ctx} do
    ir = ir(ctx)

    for strategy <- [:worker, :dirty, :auto] do
      assert {{:ok, ^ir}, receipt} =
               Profile.capture(&MLIR.verify(ir, strategy: strategy, profile: &1))

      costs = receipt.native["verify/#{if strategy == :auto, do: :dirty, else: strategy}"]
      assert receipt.phases.verify.count == costs.count and costs.count == 1
      assert costs.callback_count == 0 and costs.duration_ns > 0

      for key <- [:duration_ns, :queue_wait_ns, :completion_delivery_ns],
          do: assert(costs[key] in 0..receipt.phases.verify.duration_ns)

      assert receipt.duration_ns > 0 and receipt.vm_cpu_time_ns >= 0
      assert receipt.vm_peak_rss_after_bytes >= receipt.vm_peak_rss_before_bytes
      assert JSON.decode!(JSON.encode!(receipt))["schema_version"] == 1
    end

    break_signature(function(ir))

    errors =
      for strategy <- [:worker, :dirty, :auto] do
        assert {:error, diagnostics} = MLIR.verify(ir, strategy: strategy)
        MLIR.Diagnostic.format(diagnostics)
      end

    assert length(Enum.uniq(errors)) == 1
    assert hd(errors) =~ "entry block"
  end

  test "automatic verification retains callbacks for dynamic traits", %{ctx: ctx} do
    Beaver.Slang.load(ctx, CustomTraitSlang)
    ir = ir(ctx, "module { \"custom_trait_test.valid\"() : () -> () }")
    assert {{:ok, ^ir}, receipt} = Profile.capture(&MLIR.verify(ir, strategy: :auto, profile: &1))
    assert receipt.native["verify/worker"].callback_count > 0

    assert_raise Kinda.CallError, ~r/CallbackContextRequiresWorker/, fn ->
      MLIR.verify(ir, strategy: :dirty)
    end
  end

  test "conversion contributes callback costs to the shared receipt", %{ctx: ctx} do
    ir = ir(ctx)

    Native.with_resource(
      MLIR.ConversionTarget.create(ctx),
      &MLIR.ConversionTarget.destroy/1,
      fn target ->
        for name <- ["builtin", "arith"],
            do: MLIR.ConversionTarget.add_legal_dialect(target, name)

        MLIR.ConversionTarget.add_legal_op(target, "func.return")
        MLIR.ConversionTarget.add_dynamically_legal_op(target, "func.func", fn _ -> :legal end)

        assert {{:ok, ^ir, []}, receipt} =
                 Profile.capture(fn p ->
                   MLIR.Conversion.apply(:full, ir, target, MLIR.RewritePatternSet.create(ctx),
                     profile: p
                   )
                 end)

        costs = receipt.native["conversion/worker"]
        assert costs.callback_count > 0 and costs.callback_wait_ns > 0
        assert costs.duration_ns <= receipt.phases.conversion.duration_ns
        assert costs.queue_wait_ns <= receipt.phases.conversion.duration_ns
      end
    )
  end

  test "BEAM passes retain lifecycle state, caller ownership and deterministic cleanup", %{
    ctx: ctx
  } do
    ir = functions(ctx)
    assert ^ir = nested(ir, Lifecycle) |> Composer.run!(execution: :beam)

    for message <- [
          {:construct, nil},
          {:construct, 0},
          {:run, "f1", 0},
          {:run, "f2", 1},
          {:destruct, 2},
          {:destruct, :initialized}
        ],
        do: assert_received(^message)

    assert Context.thread_safe?(ctx)

    assert Native.to_term(CAPI.beaverContextIsMultithreadingEnabled(ctx)) ==
             not Context.independent_locking_supported?()
  end

  test "prepared pipelines isolate state across concurrent runs", %{ctx: ctx} do
    plan =
      Composer.new(ctx: ctx)
      |> Composer.nested("func.func", [Lifecycle, "canonicalize"])
      |> Composer.prepare()

    inputs = for _ <- 1..8, do: ir(ctx)

    concurrently(inputs, fn ir ->
      assert {:ok, ^ir, []} = Composer.Prepared.run(plan, ir)
      assert_received {:run, "f", 0}
      refute_received {:run, "f", 1}
      assert_received {:destruct, 1}
      assert_received {:destruct, :initialized}
    end)
  end

  test "mixed native and BEAM stages retain per-operation order", %{ctx: ctx} do
    ir = functions(ctx)

    observer = fn phase ->
      {Atom.to_string(phase), "func.func",
       fn op ->
         send(
           self(),
           {phase, MLIR.Attribute.value(op[:sym_name]), MLIR.to_string(op) =~ "arith.addi"}
         )
       end}
    end

    nested(ir, [observer.(:before), "canonicalize", observer.(:after)])
    |> Composer.run!(execution: :beam)

    for message <- [
          {:before, "f1", true},
          {:after, "f1", false},
          {:before, "f2", true},
          {:after, "f2", false}
        ],
        do: assert_receive(^message)

    assert MLIR.verify?(ir)
  end

  test "BEAM failures stop later stages and clean initialized and cloned state", %{ctx: ctx} do
    ir = functions(ctx)

    assert {:error, diagnostics} =
             nested(ir, [Failing, Lifecycle]) |> Composer.run(execution: :beam)

    assert MLIR.Diagnostic.format(diagnostics) =~ "beam pass failed"

    for message <- [{:failed_destruct, :cloned}, {:failed_destruct, :initialized}, {:destruct, 0}],
        do: assert_received(^message)

    refute_received {:run, _, _}
    assert {:error, _} = Composer.append(ir, InitFailure) |> Composer.run(execution: :beam)
    assert_received {:init_destruct, :failed_init}
  end

  test "nested native and BEAM pipelines verify parent symbol invariants", %{ctx: ctx} do
    pass =
      {"rename", "func.func", &attr(&1, "sym_name", MLIR.Attribute.string("duplicate", ctx: ctx))}

    for execution <- [:beam, :native] do
      assert {:error, diagnostics} =
               functions(ctx) |> nested(pass) |> Composer.run(execution: execution)

      assert MLIR.Diagnostic.format(diagnostics) =~ "redefinition of symbol"
    end
  end

  test "pipeline verification detects invalid IR at the selected boundary", %{ctx: ctx} do
    for verifier <- [true, :pipeline] do
      observe = {"observe", "func.func", fn _ -> send(self(), {:after_invalid, verifier}) end}
      passes = [{"change-signature", "func.func", &break_signature/1}, observe]

      assert {:error, diagnostics} =
               functions(ctx, 1)
               |> nested(passes)
               |> Composer.run(execution: :beam, verifier: verifier)

      assert MLIR.Diagnostic.format(diagnostics) =~ "entry block"

      if verifier == :pipeline,
        do: assert_received({:after_invalid, :pipeline}),
        else: refute_received({:after_invalid, true})
    end
  end

  test "native prefilters reject before callbacks and recognize properties", %{ctx: ctx} do
    ir = functions(ctx)
    rejected = [Filtered.wrong_arity(), Filtered.missing_attribute(), Filtered.missing_result()]

    property = %{
      root: "arith.constant",
      prefilter: {0, 1, ["value"]},
      match_and_rewrite: fn _, _, _, state -> {:error, state} end
    }

    for {d, expected} <- Enum.map(rejected, &{&1, 0}) ++ [{property, 2}] do
      set = MLIR.RewritePatternSet.create(ctx)

      pattern =
        MLIR.RewritePattern.create(d.root,
          ctx: ctx,
          prefilter: d.prefilter,
          match_and_rewrite: d.match_and_rewrite
        )

      MLIR.RewritePatternSet.add(set, pattern)

      Native.with_resource(
        MLIR.RewritePatternSet.freeze(set),
        &MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, &1),
        fn frozen ->
          Rewrite.apply_patterns!(ir, frozen, enable_folding: false)
          count = MLIR.RewritePattern.callback_count(pattern)
          if expected == 0, do: assert(count == 0), else: assert(count >= expected)
        end
      )
    end
  end

  test "batch snapshots preserve values and invalid edits fail before mutation", %{ctx: ctx} do
    ir = ir(ctx)
    [op, ret] = operations(ir)
    [snapshot] = IRBatch.read([op], attributes: [:note])
    assert snapshot.name == "arith.ori" and snapshot.attributes == %{"note" => nil}

    assert Enum.zip(snapshot.operands, Walker.operands(op))
           |> Enum.all?(fn {a, b} -> MLIR.equal?(a, b) end)

    assert length(snapshot.result_types) == 1
    note = MLIR.Attribute.string("batched", ctx: ctx)

    assert {{:ok, _, []}, receipt} =
             Profile.capture(fn p ->
               IRBatch.update(
                 ir,
                 [{:attribute, op, :note, note}, {:operand, ret, 0, hd(snapshot.operands)}],
                 profile: p
               )
             end)

    assert receipt.phases.batch_update.count == 1
    assert MLIR.Attribute.value(op[:note]) == "batched"
    assert MLIR.equal?(CAPI.mlirOperationGetOperand(ret, 0), hd(snapshot.operands))

    assert_raise ArgumentError, ~r/InvalidOperandIndex/, fn ->
      IRBatch.update(ir, [
        {:attribute, op, :note, MLIR.Attribute.string("wrong", ctx: ctx)},
        {:operand, ret, 50, hd(snapshot.operands)}
      ])
    end

    assert MLIR.Attribute.value(op[:note]) == "batched"
    assert_raise ArgumentError, ~r/BatchTooLarge/, fn -> IRBatch.read(List.duplicate(op, 257)) end
    other = context()

    assert_raise ArgumentError, ~r/ContextMismatch/, fn ->
      IRBatch.update(ir, [
        {:attribute, op, :local, note},
        {:attribute, op, :foreign, MLIR.Attribute.unit(ctx: other)}
      ])
    end

    assert :error = Operation.fetch(op, "local")
  end

  test "declarative replacement agrees with callbacks and avoids BEAM calls", %{ctx: ctx} do
    first = ir(ctx)
    second = ir(ctx)

    program =
      Program.new("arith.ori",
        operands: 2,
        results: 1,
        same_operands: [{0, 1}],
        operand_types: [{0, MLIR.Type.i64(ctx: ctx)}],
        rewrite: {:operand, 0}
      )

    {{result, []}, receipt} = Rewrite.profile(first, [program], enable_folding: false)

    assert MLIR.LogicalResult.success?(result) and
             receipt.native["rewrite/worker"].callback_count == 0

    callback =
      {"arith.ori",
       fn _, op, rewriter, state ->
         [a, b] = Enum.to_list(Walker.operands(op))

         if MLIR.equal?(a, b) do
           MLIR.RewriterBase.replace_op(MLIR.PatternRewriter.as_base(rewriter), op, [a])
           {:ok, state}
         else
           {:error, state}
         end
       end}

    Rewrite.apply_patterns!(second, [callback], enable_folding: false)

    assert Operation.equivalent?(Operation.from_module(first), Operation.from_module(second),
             ignore_commutativity: true
           )

    assert MLIR.verify?(first)
  end

  @alloc "module { llvm.func @malloc(i64) -> !llvm.ptr llvm.func @free(!llvm.ptr) llvm.func @enif_alloc(i64) -> !llvm.ptr llvm.func @enif_free(!llvm.ptr) llvm.func @f(%n: i64) -> !llvm.ptr { %p = llvm.call @malloc(%n) : (i64) -> !llvm.ptr llvm.call @free(%p) : (!llvm.ptr) -> () llvm.return %p : !llvm.ptr } }"
  test "property edits converge and frozen native programs can run concurrently", %{ctx: ctx} do
    old = MLIR.Attribute.flat_symbol_ref("malloc", ctx: ctx)
    new = MLIR.Attribute.flat_symbol_ref("enif_alloc", ctx: ctx)

    program =
      Program.new("llvm.call", attributes: [callee: old], rewrite: {:attributes, [callee: new]})

    inputs = for _ <- 1..6, do: ir(ctx, @alloc)

    Rewrite.with_patterns(ctx, [program], fn frozen ->
      concurrently(inputs, fn ir ->
        Rewrite.apply_patterns!(ir, frozen, enable_folding: false)
        assert MLIR.to_string(ir) =~ "llvm.call @enif_alloc"
        refute MLIR.to_string(ir) =~ "llvm.call @malloc"
        assert MLIR.verify?(ir)
      end)
    end)
  end

  test "allocation PDL rewrites agree across native and BEAM drivers", %{ctx: ctx} do
    inputs =
      for execution <- [:native, :beam] do
        ir = ir(ctx, @alloc)

        Composer.nested(ir, "llvm.func", Beaver.ENIF.UseENIFAlloc)
        |> Composer.run!(execution: execution)

        for symbol <- ["enif_alloc", "enif_free"],
            do: assert(MLIR.to_string(ir) =~ "llvm.call @#{symbol}")

        for symbol <- ["malloc", "free"], do: refute(MLIR.to_string(ir) =~ "llvm.call @#{symbol}")
        Operation.from_module(ir)
      end

    assert Operation.equivalent?(hd(inputs), List.last(inputs), ignore_commutativity: true)
  end

  test "declarative mismatches and idempotent edits converge", %{ctx: ctx} do
    tag = MLIR.Attribute.string("done", ctx: ctx)

    assert_raise ArgumentError, ~r/duplicate rewrite attribute/, fn ->
      Program.new("arith.ori", rewrite: {:attributes, [tag: tag, tag: tag]})
    end

    ir =
      ir(
        ctx,
        "module { func.func @f(%x: i64, %y: i64) -> i64 { %v = arith.ori %x, %y : i64 return %v : i64 } }"
      )

    patterns = [
      Program.new("arith.ori", same_operands: [{0, 1}], rewrite: {:operand, 0}),
      Program.new("arith.ori",
        result_types: [{0, MLIR.Type.f32(ctx: ctx)}],
        rewrite: {:operand, 0}
      ),
      Program.new("arith.ori", rewrite: :erase),
      Program.new("arith.ori", rewrite: {:attributes, [tag: tag]})
    ]

    Rewrite.apply_patterns!(ir, patterns, enable_folding: false)
    [op, _] = operations(ir)
    assert Operation.name(op) == "arith.ori" and MLIR.Attribute.value(op[:tag]) == "done"
    assert MLIR.verify?(ir)
  end

  test "one admission slot supports nested pattern callbacks in both pool modes", %{ctx: ctx} do
    for ctx <- [ctx, context(threading: true, thread_pool: nil)] do
      parent = self()

      pass =
        {"nested", "func.func",
         fn op ->
           Rewrite.apply_patterns!(
             op,
             [
               {"arith.addi",
                fn _, _, _, state ->
                  send(parent, :pattern_called)
                  {:error, state}
                end}
             ],
             enable_folding: false
           )
         end}

      assert {{:ok, _, []}, receipt} = functions(ctx) |> nested(pass) |> Composer.profile()
      assert_receive :pattern_called
      assert_receive :pattern_called
      assert receipt.native["pass/worker"].callback_count > 0
      assert Context.executor_stats(ctx).peak_running == 1
    end
  end

  test "post-parse verification in an LLVM-owned pool submits nested work" do
    ctx = context(threading: true, thread_pool: nil)
    plain = ir(ctx, "module {}")
    Beaver.Slang.load(ctx, CustomTraitSlang)

    MLIR.Trait.attach_custom(ctx, "custom_trait_test.valid", __MODULE__.NestedVerifier,
      verify: fn _ ->
        {:ok, _} = MLIR.verify(plain)
        :ok
      end
    )

    source = "module { func.func @f() { \"custom_trait_test.valid\"() : () -> () return } }"

    task =
      Task.async(fn ->
        Native.with_resource(
          MLIR.Module.create!(source, ctx: ctx),
          &MLIR.Module.destroy/1,
          &MLIR.verify?/1
        )
      end)

    assert Task.await(task, 5000)
  end

  test "nested adaptor cloning can submit work at capacity one" do
    ctx = context(threading: true, thread_pool: nil)
    plain = ir(ctx, "module {}")
    parent = self()

    source =
      "module {" <>
        Enum.map_join(
          1..3,
          &"module @m#{&1} { func.func @f() { return } func.func @g() { return } }"
        ) <> "}"

    pass =
      MLIR.Pass.create("nested-clone", "nested clone", "func.func",
        ctx: ctx,
        clone: fn state ->
          {:ok, _} = MLIR.verify(plain)
          send(parent, :cloned)
          state
        end
      )

    assert {{:ok, _, []}, receipt} =
             Composer.nested(ir(ctx, source), "builtin.module", [{"func.func", [pass]}])
             |> Composer.profile()

    assert_received :cloned
    assert receipt.native["pass/worker"].callback_count > 6
    assert Context.executor_stats(ctx).peak_running == 1
  end

  test "blocked callbacks admit short compilations, reject teardown and retire idle workers", %{
    ctx: ctx
  } do
    long_ir = ir(ctx)
    inputs = for _ <- 1..8, do: ir(ctx)
    parent = self()

    long =
      Task.async(fn ->
        Composer.append(
          long_ir,
          {"wait", "builtin.module",
           fn _ ->
             send(parent, {:waiting, self()})
             receive do: (:resume -> :ok)
           end}
        )
        |> Composer.run!()
      end)

    assert_receive {:waiting, callback}, 5000

    try do
      assert_raise ArgumentError, ~r/active callback executor work/, fn ->
        Context.destroy(ctx)
      end

      concurrently(inputs, fn small ->
        assert {:ok, ^small, []} =
                 Composer.append(small, CAPI.beaverCreateNoopPass())
                 |> Composer.append("canonicalize")
                 |> Composer.run()
      end)
    after
      send(callback, :resume)
      Task.await(long, 5000)
    end

    Process.sleep(100)
    assert %{threads: 0, running: 0, queued: 0, peak_running: 1} = Context.executor_stats(ctx)
  end
end
