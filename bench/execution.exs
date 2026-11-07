# MIX_ENV=prod mix run bench/execution.exs [operations=1000] [samples=100] [hit_rate=100]
# Use an optimized native build for decisions; Debug remains useful for regressions.
defmodule ExecutionBench do
  alias Beaver.{Composer, MLIR, Walker}
  alias Beaver.MLIR.{CAPI, Context, IRBatch, Operation, Rewrite}
  alias Beaver.Pattern.Native.Program

  def run(count, samples, hit_rate) do
    ctx = context(4)

    try do
      empty = "module {" <> Enum.map_join(1..count, &"func.func @f#{&1}() { return }") <> "}"
      pass = {"noop", "func.func", fn _ -> :ok end}

      prepared =
        Composer.new(ctx: ctx) |> Composer.nested("func.func", pass) |> Composer.prepare()

      drivers = [
        native_elixir: [],
        native_cpp: [],
        beam_worker: [strategy: :worker],
        beam_dirty: [strategy: :dirty],
        beam_auto: [strategy: :auto],
        prepared_auto: [strategy: :auto]
      ]

      passes =
        for {driver, opts} <- drivers, verifier <- [false, true], into: %{} do
          {"#{driver}/verify_#{verifier}",
           fn ->
             with_ir(ctx, empty, fn ir ->
               timed(fn ->
                 run_pass(driver, ir, pass, prepared, verifier, opts)
               end)
             end)
           end}
        end
        |> sample_cases(samples)

      verification =
        for strategy <- [:worker, :dirty, :auto], into: %{} do
          {strategy,
           fn ->
             with_ir(ctx, empty, fn ir ->
               ops = ir |> MLIR.Module.body() |> Walker.operations() |> Enum.to_list()

               timed(fn ->
                 Enum.each(ops, fn op -> {:ok, _} = MLIR.verify(op, strategy: strategy) end)
               end)
             end)
           end}
        end
        |> sample_cases(samples)

      adds =
        Enum.map_join(1..count, fn n ->
          input = if n == 1, do: "%x", else: "%v#{n - 1}"
          "%v#{n} = arith.addi #{input}, %x : i64\n"
        end)

      source = "module { func.func @f(%x: i64) -> i64 { #{adds} return %v#{count} : i64 } }"

      reads =
        for size <- [1, 32, 128, 256], into: %{} do
          {if(size == 1, do: "scalar", else: "batch_#{size}"),
           fn ->
             with_ir(ctx, source, fn ir ->
               ops = operations(ir)

               timed(fn ->
                 rows =
                   if size == 1,
                     do: Enum.map(ops, &scalar_snapshot/1),
                     else:
                       Enum.flat_map(
                         Enum.chunk_every(ops, size),
                         &IRBatch.read(&1, attributes: [:tag])
                       )

                 ^count = length(rows)
               end)
             end)
           end}
        end
        |> sample_cases(samples)

      writes =
        for size <- [1, 32, 128, 256], into: %{} do
          {if(size == 1, do: "scalar", else: "batch_#{size}"),
           fn ->
             with_ir(ctx, source, fn ir ->
               ops = operations(ir)
               attribute = MLIR.Attribute.string("bench", ctx: ctx)

               timed(fn ->
                 if size == 1 do
                   MLIR.IRRewriter.with_rewriter(ctx, fn rewriter ->
                     Enum.each(ops, fn op ->
                       CAPI.mlirRewriterBaseStartOpModification(rewriter, op)

                       CAPI.mlirOperationSetAttributeByName(
                         op,
                         MLIR.StringRef.create("tag"),
                         attribute
                       )

                       CAPI.mlirRewriterBaseFinalizeOpModification(rewriter, op)
                     end)
                   end)
                 else
                   Enum.each(Enum.chunk_every(ops, size), fn group ->
                     {:ok, _, []} =
                       IRBatch.update(ir, Enum.map(group, &{:attribute, &1, "tag", attribute}),
                         verifier: false
                       )
                   end)
                 end

                 {:ok, _} = MLIR.verify(ir, strategy: :dirty)
               end)
             end)
           end}
        end
        |> sample_cases(samples)

      {rewrites, costs} = rewrites(ctx, count, samples, hit_rate)

      cold =
        for strategy <- [:worker, :auto], into: %{} do
          {strategy,
           fn ->
             timed(fn ->
               with_ir(ctx, empty, fn ir ->
                 Composer.nested(ir, "func.func", pass)
                 |> Composer.run!(execution: :beam, verify_execution: strategy)
               end)
             end)
           end}
        end
        |> sample_cases(samples)

      profile =
        with_ir(ctx, empty, fn ir ->
          {{:ok, _, []}, receipt} =
            Composer.nested(ir, "func.func", pass)
            |> Composer.profile(execution: :beam, verify_execution: :auto)

          receipt
        end)

      IO.puts(
        JSON.encode!(
          json(%{
            metadata: %{
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
            },
            passes: passes,
            verifier_only: verification,
            reads: reads,
            writes: writes,
            rewrites: rewrites,
            cold_parse_run_destroy: cold,
            profile: profile,
            rewrite_profile: costs,
            mixed:
              for(
                capacity <- [1, 4],
                into: %{},
                do: {"capacity_#{capacity}", mixed(capacity, max(samples, 100))}
              )
          })
        )
      )
    after
      Context.destroy(ctx)
    end
  end

  defp run_pass(:native_cpp, ir, _, _, verifier, _) do
    Composer.nested(ir, "func.func", CAPI.beaverCreateNoopPass())
    |> Composer.run!(verifier: verifier)
  end

  defp run_pass(:native_elixir, ir, pass, _, verifier, _) do
    Composer.nested(ir, "func.func", pass) |> Composer.run!(verifier: verifier)
  end

  defp run_pass(:prepared_auto, ir, _, prepared, verifier, opts) do
    Composer.Prepared.run!(prepared, ir, verifier: verifier, verify_execution: opts[:strategy])
  end

  defp run_pass(_, ir, pass, _, verifier, opts) do
    Composer.nested(ir, "func.func", pass)
    |> Composer.run!(execution: :beam, verifier: verifier, verify_execution: opts[:strategy])
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
      Rewrite.with_patterns(ctx, [callback], fn callback_frozen ->
        Rewrite.with_patterns(ctx, [program], fn native_frozen ->
          for {engine, patterns, frozen} <- [
                {:callback, [callback], callback_frozen},
                {:declarative, [program], native_frozen}
              ],
              {phase, set} <- [warm: frozen, prepare_apply_destroy: patterns],
              into: %{} do
            {{engine, phase},
             fn ->
               with_ir(ctx, source, fn ir ->
                 duration =
                   timed(fn -> Rewrite.apply_patterns!(ir, set, enable_folding: false) end)

                 true = MLIR.verify?(ir)
                 ^count = length(Regex.scan(~r/llvm.call @enif_alloc/, MLIR.to_string(ir)))
                 duration
               end)
             end}
          end
          |> sample_cases(samples)
          |> Enum.reduce(%{}, fn {{engine, phase}, stats}, acc ->
            Map.update(acc, engine, %{phase => stats}, &Map.put(&1, phase, stats))
          end)
        end)
      end)

    costs =
      with_ir(ctx, source, fn ir ->
        {{result, []}, receipt} = Rewrite.profile(ir, [program], enable_folding: false)
        true = MLIR.LogicalResult.success?(result)
        receipt
      end)

    {cases, costs}
  end

  defp mixed(capacity, jobs) do
    ctx = context(capacity)
    parent = self()

    try do
      with_ir(ctx, "module {}", fn long_ir ->
        long =
          Task.async(fn ->
            Composer.append(
              long_ir,
              {"blocked", "builtin.module",
               fn _ ->
                 send(parent, {:blocked, self()})

                 receive do
                   :resume -> :ok
                 end
               end}
            )
            |> Composer.run!()
          end)

        callback =
          receive do
            {:blocked, pid} -> pid
          end

        try do
          elapsed = System.monotonic_time(:microsecond)

          latencies =
            1..jobs
            |> Task.async_stream(
              fn _ ->
                timed(fn ->
                  with_ir(ctx, "module {}", fn ir ->
                    Composer.append(ir, CAPI.beaverCreateNoopPass()) |> Composer.run!()
                  end)
                end)
              end,
              max_concurrency: 8,
              timeout: 30_000
            )
            |> Enum.map(fn {:ok, us} -> us end)

          wall = System.monotonic_time(:microsecond) - elapsed

          Map.merge(summary(latencies), %{
            jobs_per_second: jobs * 1_000_000 / wall,
            executor: Context.executor_stats(ctx)
          })
        after
          send(callback, :resume)
          Task.await(long, 30_000)
        end
      end)
    after
      Context.destroy(ctx)
    end
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
      attributes: %{
        "tag" =>
          case Operation.fetch(op, "tag") do
            {:ok, attr} -> attr
            :error -> nil
          end
      }
    }
  end

  defp operations(ir),
    do:
      ir
      |> MLIR.Module.body()
      |> Walker.operations()
      |> Enum.at(0)
      |> MLIR.Dialect.Func.entry_block()
      |> Walker.operations()
      |> Enum.filter(&(Operation.name(&1) == "arith.addi"))

  defp context(capacity),
    do:
      Context.create(
        threading: false,
        thread_safe: true,
        callback_executor: [max_concurrency: capacity]
      )

  defp with_ir(ctx, source, fun) do
    ir = MLIR.Module.create!(source, ctx: ctx)

    try do
      fun.(ir)
    after
      MLIR.Module.destroy(ir)
    end
  end

  defp timed(fun), do: :timer.tc(fun) |> elem(0)

  defp sample_cases(cases, n) do
    cases = Enum.sort(cases)
    Enum.each(cases, fn {_, fun} -> fun.() end)

    values =
      Enum.reduce(0..(n - 1), %{}, fn round, acc ->
        {first, last} = Enum.split(cases, rem(round, length(cases)))

        Enum.reduce(last ++ first, acc, fn {key, fun}, result ->
          elapsed = fun.()
          Map.update(result, key, [elapsed], fn samples -> [elapsed | samples] end)
        end)
      end)

    Map.new(values, fn {key, samples} -> {key, summary(Enum.reverse(samples))} end)
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

  defp json(map) when is_map(map),
    do:
      Map.new(map, fn {k, v} ->
        {if(is_tuple(k), do: k |> Tuple.to_list() |> Enum.join("/"), else: k), json(v)}
      end)

  defp json(list) when is_list(list), do: Enum.map(list, &json/1)
  defp json(value), do: value
end

[count, samples, hit_rate] =
  Enum.with_index([1000, 100, 100])
  |> Enum.map(fn {default, i} ->
    case Enum.at(System.argv(), i) do
      nil -> default
      arg -> String.to_integer(arg)
    end
  end)

if count < 1 or samples < 1 or hit_rate not in 0..100,
  do: raise(ArgumentError, "positive counts and a hit rate in 0..100 are required")

ExecutionBench.run(count, samples, hit_rate)
