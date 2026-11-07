defmodule ExecutionOptimizationTest do
  use ExUnit.Case, async: true
  alias Beaver.{Composer, MLIR, Walker}
  alias Beaver.Execution.Profile
  alias Beaver.MLIR.{CAPI, Context, IRBatch, Operation, Rewrite}
  alias Beaver.Pattern.Native.Program

  setup do
    ctx =
      Context.create(
        threading: not Context.independent_locking_supported?(),
        thread_safe: true,
        callback_executor: [max_concurrency: 1, idle_timeout_ms: 25]
      )

    on_exit(fn -> Context.destroy(ctx) end)
    %{ctx: ctx}
  end

  defp with_ir(
         ctx,
         fun,
         source \\ "module { func.func @f(%x: i64) -> i64 { %v = arith.ori %x, %x : i64 return %v : i64 } }"
       ) do
    ir = MLIR.Module.create!(source, ctx: ctx)

    try do
      fun.(ir)
    after
      MLIR.Module.destroy(ir)
    end
  end

  defp function(ir), do: ir |> MLIR.Module.body() |> Walker.operations() |> Enum.at(0)

  defp operations(ir),
    do:
      ir |> function() |> MLIR.Dialect.Func.entry_block() |> Walker.operations() |> Enum.to_list()

  test "verifier strategies preserve diagnostics and attribute costs per run", %{ctx: ctx} do
    with_ir(ctx, fn ir ->
      for strategy <- [:worker, :dirty, :auto] do
        assert {{:ok, ^ir}, receipt} =
                 Profile.capture(fn profile ->
                   MLIR.verify(ir, strategy: strategy, profile: profile)
                 end)

        assert receipt.phases.verify.count == 1
        actual = if strategy == :auto, do: :dirty, else: strategy
        assert receipt.native["verify/#{actual}"].count == 1
        assert receipt.native["verify/#{actual}"].callback_count == 0
        assert receipt.native["verify/#{actual}"].duration_ns > 0
        assert receipt.native["verify/#{actual}"].duration_ns <= receipt.phases.verify.duration_ns
        assert receipt.native["verify/#{actual}"].queue_wait_ns >= 0

        assert receipt.native["verify/#{actual}"].queue_wait_ns <=
                 receipt.phases.verify.duration_ns

        assert receipt.native["verify/#{actual}"].completion_delivery_ns <=
                 receipt.phases.verify.duration_ns

        assert receipt.duration_ns > 0
        assert receipt.vm_cpu_time_ns >= 0
        assert receipt.vm_peak_rss_after_bytes >= receipt.vm_peak_rss_before_bytes
        assert JSON.decode!(JSON.encode!(receipt))["schema_version"] == 1
      end

      CAPI.mlirBlockAddArgument(
        function(ir) |> MLIR.Dialect.Func.entry_block(),
        MLIR.Type.i64(ctx: ctx),
        MLIR.Location.unknown(ctx: ctx)
      )

      errors =
        for strategy <- [:worker, :dirty, :auto] do
          assert {:error, diagnostics} = MLIR.verify(ir, strategy: strategy)
          MLIR.Diagnostic.format(diagnostics)
        end

      assert length(Enum.uniq(errors)) == 1
      assert hd(errors) =~ "entry block"
    end)
  end

  test "automatic verification retains callback execution for dynamic traits", %{ctx: ctx} do
    Beaver.Slang.load(ctx, CustomTraitSlang)

    with_ir(
      ctx,
      fn ir ->
        assert {{:ok, ^ir}, receipt} =
                 Profile.capture(fn p -> MLIR.verify(ir, strategy: :auto, profile: p) end)

        assert receipt.native["verify/worker"].callback_count > 0

        assert_raise Kinda.CallError, ~r/CallbackContextRequiresWorker/, fn ->
          MLIR.verify(ir, strategy: :dirty)
        end
      end,
      "module { \"custom_trait_test.valid\"() : () -> () }"
    )
  end

  test "dialect conversion contributes native callback costs to the same receipt", %{ctx: ctx} do
    with_ir(ctx, fn ir ->
      target = MLIR.ConversionTarget.create(ctx)
      MLIR.ConversionTarget.add_legal_dialect(target, "builtin")
      MLIR.ConversionTarget.add_legal_dialect(target, "arith")
      MLIR.ConversionTarget.add_legal_op(target, "func.return")
      MLIR.ConversionTarget.add_dynamically_legal_op(target, "func.func", fn _ -> :legal end)

      try do
        assert {{:ok, ^ir, []}, receipt} =
                 Profile.capture(fn profile ->
                   MLIR.Conversion.apply(:full, ir, target, MLIR.RewritePatternSet.create(ctx),
                     profile: profile
                   )
                 end)

        costs = receipt.native["conversion/worker"]
        assert costs.callback_count > 0
        assert costs.callback_wait_ns > 0
        assert costs.duration_ns <= receipt.phases.conversion.duration_ns
        assert costs.queue_wait_ns <= receipt.phases.conversion.duration_ns
      after
        MLIR.ConversionTarget.destroy(target)
      end
    end)
  end

  defmodule CountingPass do
    use MLIR.Pass, on: "func.func"
    def initialize(_, _), do: {:ok, 0}

    def run(_, n),
      do:
        (
          send(self(), {:count, n})
          n + 1
        )

    def destruct(n), do: send(self(), {:destroyed, n})
  end

  test "prepared pipelines have independent state across concurrent runs", %{ctx: ctx} do
    plan =
      Composer.new(ctx: ctx)
      |> Composer.nested("func.func", [CountingPass, "canonicalize"])
      |> Composer.prepare()

    results =
      1..8
      |> Task.async_stream(
        fn _ ->
          with_ir(ctx, fn ir ->
            assert {:ok, ^ir, []} = Composer.Prepared.run(plan, ir, verify_execution: :auto)
            assert_receive {:count, 0}
            refute_received {:count, 1}
            assert_receive {:destroyed, 1}
            assert_receive {:destroyed, 0}
          end)
        end,
        max_concurrency: 4
      )
      |> Enum.to_list()

    assert Enum.all?(results, &match?({:ok, _}, &1))
  end

  test "nested BEAM and native pipelines both verify parent symbol invariants", %{ctx: ctx} do
    pass =
      {"rename", "func.func",
       fn op ->
         CAPI.mlirOperationSetAttributeByName(
           op,
           MLIR.StringRef.create("sym_name"),
           MLIR.Attribute.string("duplicate", ctx: ctx)
         )
       end}

    for execution <- [:beam, :native] do
      with_ir(
        ctx,
        fn ir ->
          assert {:error, diagnostics} =
                   Composer.nested(ir, "func.func", pass) |> Composer.run(execution: execution)

          assert MLIR.Diagnostic.format(diagnostics) =~ "redefinition of symbol"
        end,
        "module { func.func @f() { return } func.func @g() { return } }"
      )
    end
  end

  test "batch snapshots and edits preserve values and reject the entire invalid batch", %{
    ctx: ctx
  } do
    with_ir(ctx, fn ir ->
      [op, ret] = operations(ir)
      [snapshot] = IRBatch.read([op], attributes: [:note])
      assert snapshot.name == "arith.ori"
      assert snapshot.attributes == %{"note" => nil}

      assert Enum.zip(snapshot.operands, Enum.to_list(Walker.operands(op)))
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

      assert_raise ArgumentError, ~r/BatchTooLarge/, fn ->
        IRBatch.read(List.duplicate(op, 257))
      end
    end)
  end

  test "batch preflight rejects cross-context attributes without mutation", %{ctx: ctx} do
    other = Context.create()

    try do
      with_ir(ctx, fn ir ->
        [op, _] = operations(ir)

        assert_raise ArgumentError, ~r/ContextMismatch/, fn ->
          IRBatch.update(ir, [
            {:attribute, op, :local, MLIR.Attribute.unit(ctx: ctx)},
            {:attribute, op, :foreign, MLIR.Attribute.unit(ctx: other)}
          ])
        end

        assert :error = Operation.fetch(op, "local")
      end)
    after
      Context.destroy(other)
    end
  end

  test "declarative replacement agrees with a callback and produces zero callbacks", %{ctx: ctx} do
    program =
      Program.new("arith.ori",
        operands: 2,
        results: 1,
        same_operands: [{0, 1}],
        operand_types: [{0, MLIR.Type.i64(ctx: ctx)}],
        rewrite: {:operand, 0}
      )

    with_ir(ctx, fn first ->
      with_ir(ctx, fn second ->
        Rewrite.with_patterns(ctx, [program], fn frozen ->
          {{result, []}, receipt} = Rewrite.profile(first, frozen, enable_folding: false)
          assert MLIR.LogicalResult.success?(result)
          assert receipt.native["rewrite/worker"].callback_count == 0
        end)

        Rewrite.apply_patterns!(
          second,
          [
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
          ],
          enable_folding: false
        )

        assert Operation.equivalent?(Operation.from_module(first), Operation.from_module(second),
                 ignore_commutativity: true
               )

        assert MLIR.verify?(first)
      end)
    end)
  end

  test "declarative property edits converge and frozen programs are reusable", %{ctx: ctx} do
    old = MLIR.Attribute.flat_symbol_ref("malloc", ctx: ctx)
    new = MLIR.Attribute.flat_symbol_ref("enif_alloc", ctx: ctx)

    program =
      Program.new("llvm.call", attributes: [callee: old], rewrite: {:attributes, [callee: new]})

    source =
      "module { llvm.func @malloc(i64) -> !llvm.ptr llvm.func @enif_alloc(i64) -> !llvm.ptr llvm.func @f(%n: i64) -> !llvm.ptr { %p = llvm.call @malloc(%n) : (i64) -> !llvm.ptr llvm.return %p : !llvm.ptr } }"

    Rewrite.with_patterns(ctx, [program], fn frozen ->
      tasks =
        for _ <- 1..6 do
          Task.async(fn ->
            with_ir(
              ctx,
              fn ir ->
                Rewrite.apply_patterns!(ir, frozen, enable_folding: false)
                text = MLIR.to_string(ir)
                assert text =~ "llvm.call @enif_alloc"
                refute text =~ "llvm.call @malloc"
                assert MLIR.verify?(ir)
              end,
              source
            )
          end)
        end

      Enum.each(tasks, &Task.await(&1, 5000))
    end)
  end

  test "declarative constraints reject mismatches and idempotent edits converge", %{ctx: ctx} do
    tag = MLIR.Attribute.string("done", ctx: ctx)

    assert_raise ArgumentError, ~r/duplicate rewrite attribute/, fn ->
      Program.new("arith.ori", rewrite: {:attributes, [tag: tag, tag: tag]})
    end

    with_ir(
      ctx,
      fn ir ->
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
        assert Operation.name(op) == "arith.ori"
        assert MLIR.Attribute.value(op[:tag]) == "done"
        assert MLIR.verify?(ir)
      end,
      "module { func.func @f(%x: i64, %y: i64) -> i64 { %v = arith.ori %x, %y : i64 return %v : i64 } }"
    )
  end

  test "one admission slot supports nested callbacks with MLIR parallel execution" do
    ctx =
      Context.create(
        thread_pool: nil,
        callback_executor: [max_concurrency: 1, idle_timeout_ms: 25]
      )

    try do
      with_ir(
        ctx,
        fn ir ->
          pass =
            {"nested", "func.func",
             fn op ->
               Rewrite.apply_patterns!(
                 op,
                 [{"arith.ori", fn _, _, _, state -> {:error, state} end}],
                 enable_folding: false
               )
             end}

          assert {{:ok, ^ir, []}, receipt} =
                   Composer.nested(ir, "func.func", pass) |> Composer.profile()

          assert receipt.native["pass/worker"].callback_count > 0
          assert Context.executor_stats(ctx).peak_running == 1
        end,
        "module { func.func @a(%x: i64) -> i64 { %v = arith.ori %x, %x : i64 return %v : i64 } func.func @b(%x: i64) -> i64 { %v = arith.ori %x, %x : i64 return %v : i64 } }"
      )
    after
      Context.destroy(ctx)
    end
  end

  test "post-parse verification in an LLVM-owned pool can submit nested work" do
    ctx = Context.create(thread_pool: nil, callback_executor: [max_concurrency: 1])
    plain = MLIR.Module.create!("module {}", ctx: ctx)

    try do
      Beaver.Slang.load(ctx, CustomTraitSlang)

      MLIR.Trait.attach_custom(ctx, "custom_trait_test.valid", __MODULE__.NestedVerifier,
        verify: fn _ ->
          {:ok, _} = MLIR.verify(plain)
          :ok
        end
      )

      source = "module { func.func @f() { \"custom_trait_test.valid\"() : () -> () return } }"
      task = Task.async(fn -> with_ir(ctx, fn ir -> assert MLIR.verify?(ir) end, source) end)
      Task.await(task, 5000)
    after
      MLIR.Module.destroy(plain)
      Context.destroy(ctx)
    end
  end

  test "nested adaptor cloning in an LLVM-owned pool can submit work at capacity one" do
    ctx = Context.create(thread_pool: nil, callback_executor: [max_concurrency: 1])
    plain = MLIR.Module.create!("module {}", ctx: ctx)

    try do
      source =
        "module {" <>
          Enum.map_join(1..3, fn n ->
            "module @m#{n} { func.func @f() { return } func.func @g() { return } }"
          end) <> "}"

      with_ir(
        ctx,
        fn ir ->
          parent = self()

          pass =
            MLIR.Pass.create("nested-clone", "nested clone", "func.func",
              ctx: ctx,
              clone: fn state ->
                {:ok, _} = MLIR.verify(plain)
                send(parent, :cloned)
                state
              end
            )

          assert {{:ok, ^ir, []}, receipt} =
                   Composer.nested(ir, "builtin.module", [{"func.func", [pass]}])
                   |> Composer.profile()

          assert_received :cloned
          assert receipt.native["pass/worker"].callback_count > 6
          assert Context.executor_stats(ctx).peak_running == 1
        end,
        source
      )
    after
      MLIR.Module.destroy(plain)
      Context.destroy(ctx)
    end
  end

  test "a long callback lends admission to short jobs and idle threads retire", %{ctx: ctx} do
    parent = self()

    with_ir(ctx, fn ir ->
      long =
        Task.async(fn ->
          Composer.append(
            ir,
            {"wait", "builtin.module",
             fn _ ->
               send(parent, {:waiting, self()})

               receive do
                 :resume -> :ok
               end
             end}
          )
          |> Composer.run!()
        end)

      assert_receive {:waiting, callback}, 5000

      try do
        assert_raise ArgumentError, ~r/active callback executor work/, fn ->
          Context.destroy(ctx)
        end

        jobs =
          for _ <- 1..8,
              do:
                Task.async(fn ->
                  with_ir(ctx, fn small ->
                    assert {:ok, ^small, []} =
                             Composer.append(small, CAPI.beaverCreateNoopPass()) |> Composer.run()
                  end)
                end)

        Enum.each(jobs, &Task.await(&1, 5000))
      after
        send(callback, :resume)
        Task.await(long, 5000)
      end
    end)

    Process.sleep(100)
    assert %{threads: 0, running: 0, queued: 0, peak_running: 1} = Context.executor_stats(ctx)
  end
end
