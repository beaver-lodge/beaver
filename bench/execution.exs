# MIX_ENV=prod mix run bench/execution.exs [operations=1000] [samples=100] [hit_rate=100]
defmodule ExecutionBench do
  alias Beaver.{Composer, MLIR, Native, Walker}
  alias Beaver.MLIR.{AnalysisSession, CAPI, Context, IRBatch, Operation, Rewrite, Transform}
  alias Beaver.Pattern.Native.Program

  def run(count, samples, hit_rate) do
    Native.with_resource(context(4), &Context.destroy/1, fn ctx ->
      empty = "module {" <> Enum.map_join(1..count, &"func.func @f#{&1}() { return }") <> "}"
      noop = {"noop", "func.func", fn _ -> :ok end}
      plan = Composer.new(ctx: ctx) |> Composer.nested("func.func", noop) |> Composer.prepare()

      drivers = ~w(native_elixir native_cpp beam_worker beam_dirty beam_auto prepared_auto)a

      passes =
        for driver <- drivers, verify <- [false, true], into: %{} do
          {"#{driver}/verify_#{verify}", scenario(ctx, empty, &run_pass(driver, &1, noop, plan, verify))}
        end

      verification =
        for strategy <- [:worker, :dirty, :auto], into: %{} do
          {strategy,
           scenario(
             ctx,
             empty,
             fn ops ->
               Enum.each(ops, fn op -> {:ok, _} = MLIR.verify(op, strategy: strategy) end)
             end,
             &top_ops/1
           )}
        end

      adds =
        Enum.map_join(1..count, fn n ->
          input = if n == 1, do: "%x", else: "%v#{n - 1}"
          "%v#{n} = arith.addi #{input}, %x : i64\n"
        end)

      source = "module { func.func @f(%x: i64) -> i64 { #{adds} return %v#{count} : i64 } }"
      attr = MLIR.Attribute.string("bench", ctx: ctx)

      batches =
        for workload <- [:reads, :writes], into: %{} do
          cases =
            for size <- [1, 32, 128, 256], into: %{} do
              {batch_name(size), scenario(ctx, source, &batch(workload, &1, size, attr), &{&1, operations(&1)})}
            end

          {workload, cases}
        end

      cold =
        for driver <- [:beam_worker, :beam_auto], into: %{} do
          {driver, fn -> timed(fn -> with_ir(ctx, empty, &run_pass(driver, &1, noop, plan, true)) end) end}
        end

      # Execute groups sequentially; each group rotates case order after warmup.
      results =
        Map.merge(batches, %{
          passes: passes,
          verifier_only: verification,
          cold_parse_run_destroy: cold
        })
        |> Map.new(fn {key, cases} -> {key, sample(cases, samples)} end)

      {rewrites, rewrite_profile} = rewrites(ctx, count, samples, hit_rate)

      {{:ok, _, []}, profile} =
        with_ir(ctx, empty, fn ir ->
          Composer.nested(ir, "func.func", noop) |> Composer.profile(execution: :beam)
        end)

      metadata = %{
        operations: count,
        samples: samples,
        hit_rate_percent: hit_rate,
        sampling: "rotating case order",
        warmup_samples_per_case: 1,
        mlir_threading: false,
        context_thread_safe: true,
        callback_executor_capacity: 4,
        elixir: System.version(),
        otp: System.otp_release(),
        schedulers: System.schedulers_online(),
        architecture: :erlang.system_info(:system_architecture) |> List.to_string(),
        native_build: CAPI.beaver_raw_execution_build_mode(),
        llvm: CAPI.beaverGetLLVMVersion() |> MLIR.to_string()
      }

      mixed = Map.new([1, 4], &{"capacity_#{&1}", mixed(&1, max(samples, 100))})

      IO.puts(
        JSON.encode!(
          Map.merge(results, %{
            metadata: metadata,
            rewrites: rewrites,
            profile: profile,
            rewrite_profile: rewrite_profile,
            analyses: analyses(ctx, source, samples),
            mixed: mixed
          })
        )
      )
    end)
  end

  defp analyses(ctx, source, samples) do
    with_ir(ctx, source, fn ir ->
      ops = operations(ir)
      first = hd(ops)
      last = List.last(ops)
      value = Operation.result(first, 0)
      queries = List.duplicate({:dominates, first, last}, 100)

      schedule = """
      module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%root: !transform.any_op) { transform.yield }
      }
      """

      {:ok, prepared} = Transform.prepare(schedule, ctx)

      Native.with_resource(prepared, &Transform.close/1, fn prepared ->
        AnalysisSession.with(ir, fn session ->
          %{
            scalar_100: fn ->
              for _ <- 1..100, do: AnalysisSession.dominates?(session, first, last)
            end,
            batch_100: fn ->
              %{results: results} = AnalysisSession.query(session, queries)
              true = Enum.all?(results)
            end,
            liveness_cold: fn ->
              AnalysisSession.with(ir, &AnalysisSession.dead_after?(&1, value, last))
            end,
            liveness_warm: fn -> AnalysisSession.dead_after?(session, value, last) end,
            transform_cold: fn -> {:ok, _} = Transform.execute(ir, schedule) end,
            transform_prepared: fn -> {:ok, _} = Transform.execute(ir, prepared) end
          }
          |> Map.new(fn {key, fun} -> {key, fn -> timed(fun) end} end)
          |> sample(samples)
        end)
      end)
    end)
  end

  defp run_pass(:prepared_auto, ir, _, plan, verify),
    do: Composer.Prepared.run!(plan, ir, verifier: verify)

  defp run_pass(driver, ir, noop, _, verify) do
    pass = if driver == :native_cpp, do: CAPI.beaverCreateNoopPass(), else: noop

    opts =
      case driver do
        :beam_worker -> [execution: :beam, verify_execution: :worker]
        :beam_dirty -> [execution: :beam, verify_execution: :dirty]
        :beam_auto -> [execution: :beam, verify_execution: :auto]
        _ -> []
      end

    Composer.nested(ir, "func.func", pass) |> Composer.run!(Keyword.put(opts, :verifier, verify))
  end

  defp rewrites(ctx, count, samples, hit_rate) do
    calls =
      Enum.map_join(1..count, fn i ->
        callee = if rem(i - 1, 100) < hit_rate, do: "malloc", else: "enif_alloc"
        "%p#{i} = llvm.call @#{callee}(%n) : (i64) -> !llvm.ptr\n"
      end)

    source =
      "module { llvm.func @malloc(i64) -> !llvm.ptr llvm.func @enif_alloc(i64) -> !llvm.ptr llvm.func @f(%n: i64) -> !llvm.ptr { #{calls} llvm.return %p#{count} : !llvm.ptr } }"

    old = MLIR.Attribute.flat_symbol_ref("malloc", ctx: ctx)
    replacement = MLIR.Attribute.flat_symbol_ref("enif_alloc", ctx: ctx)

    program =
      Program.new("llvm.call",
        attributes: [callee: old],
        rewrite: {:attributes, [callee: replacement]}
      )

    callback =
      {"llvm.call",
       fn _, op, rewriter, state ->
         if MLIR.equal?(op[:callee], old) do
           base = MLIR.PatternRewriter.as_base(rewriter)
           CAPI.mlirRewriterBaseStartOpModification(base, op)
           CAPI.mlirOperationSetAttributeByName(op, MLIR.StringRef.create("callee"), replacement)
           CAPI.mlirRewriterBaseFinalizeOpModification(base, op)
           {:ok, state}
         else
           {:error, state}
         end
       end}

    cases =
      Rewrite.with_patterns(ctx, [callback], fn cb ->
        Rewrite.with_patterns(ctx, [program], fn native ->
          for {engine, patterns, frozen} <- [
                {:callback, [callback], cb},
                {:declarative, [program], native}
              ],
              {phase, set} <- [warm: frozen, prepare_apply_destroy: patterns],
              into: %{} do
            {{engine, phase},
             scenario(
               ctx,
               source,
               &Rewrite.apply_patterns!(&1, set, enable_folding: false),
               fn ir -> ir end,
               fn ir ->
                 true = MLIR.verify?(ir)
                 ^count = length(Regex.scan(~r/llvm.call @enif_alloc/, MLIR.to_string(ir)))
               end
             )}
          end
          |> sample(samples)
          |> Enum.reduce(%{}, fn {{engine, phase}, stats}, acc ->
            Map.update(acc, engine, %{phase => stats}, &Map.put(&1, phase, stats))
          end)
        end)
      end)

    {{result, []}, receipt} =
      with_ir(ctx, source, &Rewrite.profile(&1, [program], enable_folding: false))

    true = MLIR.LogicalResult.success?(result)
    {cases, receipt}
  end

  defp mixed(capacity, jobs) do
    Native.with_resource(context(capacity), &Context.destroy/1, fn ctx ->
      with_ir(ctx, "module {}", fn ir ->
        parent = self()

        long =
          Task.async(fn ->
            Composer.append(
              ir,
              {"blocked", "builtin.module",
               fn _ ->
                 send(parent, {:blocked, self()})
                 receive do: (:resume -> :ok)
               end}
            )
            |> Composer.run!()
          end)

        callback = receive do: ({:blocked, pid} -> pid)

        try do
          started = System.monotonic_time(:microsecond)

          latencies =
            Task.async_stream(
              1..jobs,
              fn _ ->
                timed(fn ->
                  with_ir(ctx, "module {}", fn small ->
                    Composer.append(small, CAPI.beaverCreateNoopPass()) |> Composer.run!()
                  end)
                end)
              end,
              max_concurrency: 8,
              timeout: 30_000
            )
            |> Enum.map(fn {:ok, us} -> us end)

          Map.merge(summary(latencies), %{
            executor: Context.executor_stats(ctx),
            jobs_per_second: jobs * 1_000_000 / (System.monotonic_time(:microsecond) - started)
          })
        after
          send(callback, :resume)
          Task.await(long, 30_000)
        end
      end)
    end)
  end

  defp batch(:reads, {_, ops}, size, _) do
    rows =
      if size == 1,
        do: Enum.map(ops, &scalar_snapshot/1),
        else: Enum.flat_map(Enum.chunk_every(ops, size), &IRBatch.read(&1, attributes: [:tag]))

    true = length(rows) == length(ops)
  end

  defp batch(:writes, {ir, ops}, size, attr) do
    if size == 1 do
      MLIR.IRRewriter.with_rewriter(MLIR.context(ir), fn rewriter ->
        Enum.each(ops, &set_attribute(rewriter, &1, attr))
      end)
    else
      Enum.each(Enum.chunk_every(ops, size), fn group ->
        {:ok, _, []} =
          IRBatch.update(ir, Enum.map(group, &{:attribute, &1, "tag", attr}), verifier: false)
      end)
    end

    {:ok, _} = MLIR.verify(ir, strategy: :dirty)
  end

  defp scalar_snapshot(op) do
    operands = Enum.to_list(Walker.operands(op))
    results = Enum.to_list(Walker.results(op))

    %{
      operation: op,
      name: Operation.name(op),
      operands: operands,
      results: results,
      operand_types: Enum.map(operands, &MLIR.Value.type/1),
      result_types: Enum.map(results, &MLIR.Value.type/1),
      attributes: %{"tag" => op["tag"]}
    }
  end

  defp set_attribute(rewriter, op, attr) do
    CAPI.mlirRewriterBaseStartOpModification(rewriter, op)
    CAPI.mlirOperationSetAttributeByName(op, MLIR.StringRef.create("tag"), attr)
    CAPI.mlirRewriterBaseFinalizeOpModification(rewriter, op)
  end

  defp top_ops(ir), do: ir |> MLIR.Module.body() |> Walker.operations() |> Enum.to_list()

  defp operations(ir),
    do:
      ir
      |> top_ops()
      |> hd()
      |> MLIR.Dialect.Func.entry_block()
      |> Walker.operations()
      |> Enum.filter(&(Operation.name(&1) == "arith.addi"))

  defp batch_name(1), do: "scalar"
  defp batch_name(n), do: "batch_#{n}"

  defp context(n),
    do: Context.create(threading: false, thread_safe: true, callback_executor: [max_concurrency: n])

  defp with_ir(ctx, source, fun),
    do: Native.with_resource(MLIR.Module.create!(source, ctx: ctx), &MLIR.Module.destroy/1, fun)

  defp timed(fun), do: :timer.tc(fun) |> elem(0)

  defp scenario(ctx, source, run, prepare \\ fn ir -> ir end, check \\ fn _ -> :ok end) do
    fn ->
      with_ir(ctx, source, fn ir ->
        state = prepare.(ir)
        duration = timed(fn -> run.(state) end)
        check.(ir)
        duration
      end)
    end
  end

  defp sample(cases, n) do
    cases = Enum.sort(cases)
    Enum.each(cases, fn {_, fun} -> fun.() end)

    Enum.reduce(0..(n - 1), %{}, fn round, acc ->
      {first, last} = Enum.split(cases, rem(round, length(cases)))

      Enum.reduce(last ++ first, acc, fn {key, fun}, result ->
        elapsed = fun.()
        Map.update(result, key, [elapsed], &[elapsed | &1])
      end)
    end)
    |> Map.new(fn {key, values} -> {key, summary(Enum.reverse(values))} end)
  end

  defp summary(values) do
    sorted = Enum.sort(values)
    n = length(values)
    quantile = fn p -> Enum.at(sorted, max(0, ceil(n * p) - 1)) end

    %{
      samples: n,
      median_us: quantile.(0.5),
      p95_us: quantile.(0.95),
      p99_us: if(n >= 100, do: quantile.(0.99)),
      samples_us: values
    }
  end
end

[count, samples, hit_rate] =
  Enum.with_index([1000, 100, 100])
  |> Enum.map(fn {default, i} ->
    System.argv() |> Enum.at(i, to_string(default)) |> String.to_integer()
  end)

if count < 1 or samples < 1 or hit_rate not in 0..100,
  do: raise(ArgumentError, "positive counts and a hit rate in 0..100 are required")

ExecutionBench.run(count, samples, hit_rate)
