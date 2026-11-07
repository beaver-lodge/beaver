# mix run bench/callbacks.exs [operations] [repetitions]
defmodule CallbackBench do
  alias Beaver.{Composer, MLIR}
  alias Beaver.MLIR.{CAPI, Context, RewritePatternSet, StringRef}

  def run(count, repeats) do
    ctx = Context.create(threading: false, thread_safe: true)

    try do
      pass_ir =
        "module {" <> Enum.map_join(1..count, fn n -> "func.func @f#{n}() { return }" end) <> "}"

      pass = {"noop", "func.func", fn _ -> :ok end}

      passes =
        for verifier <- [false, true, :pipeline],
            execution <- [:native, :beam],
            verifier != :pipeline or execution == :beam,
            into: %{} do
          {"#{execution}_verify_#{verifier}",
           measure(repeats, fn ->
             with_module(ctx, pass_ir, fn ir ->
               elapsed(fn ->
                 Composer.nested(ir, "func.func", pass)
                 |> Composer.run!(execution: execution, verifier: verifier)
               end)
             end)
           end)}
        end

      adds =
        Enum.map_join(1..count, fn n ->
          input = if n == 1, do: "%x", else: "%v#{n - 1}"
          "%v#{n} = arith.addi #{input}, %x : i64\n"
        end)

      pattern_ir = "module { func.func @f(%x: i64) -> i64 { #{adds} return %v#{count} : i64 } }"

      filters =
        for {name, filter} <- [{"beam_filter", {-1, -1, []}}, {"native_filter", {3, -1, []}}],
            into: %{} do
          {name,
           measure(repeats, fn ->
             with_module(ctx, pattern_ir, fn ir ->
               set = RewritePatternSet.create(ctx)

               pattern =
                 MLIR.RewritePattern.create("arith.addi",
                   ctx: ctx,
                   prefilter: filter,
                   match_and_rewrite: fn _, op, _, state ->
                     2 = CAPI.mlirOperationGetNumOperands(op) |> Beaver.Native.to_term()
                     {:error, state}
                   end
                 )

               RewritePatternSet.add(set, pattern)
               frozen = RewritePatternSet.freeze(set)

               try do
                 time =
                   elapsed(fn ->
                     MLIR.Rewrite.apply_patterns!(ir, frozen, enable_folding: false)
                   end)

                 Map.put(time, "callbacks", MLIR.RewritePattern.callback_count(pattern))
               after
                 MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen)
               end
             end)
           end)}
        end

      calls =
        Enum.map_join(1..count, &"%p#{&1} = llvm.call @malloc(%size) : (i64) -> !llvm.ptr\n")

      alloc_ir = """
      module {
        llvm.func @malloc(i64) -> !llvm.ptr
        llvm.func @enif_alloc(i64) -> !llvm.ptr
        llvm.func @f(%size: i64) -> !llvm.ptr { #{calls} llvm.return %p#{count} : !llvm.ptr }
      }
      """

      alloc =
        for engine <- [:beam, :pdl], into: %{} do
          {Atom.to_string(engine),
           measure(repeats, fn ->
             with_module(ctx, alloc_ir, fn ir ->
               {set, pattern, pdl_module} = allocation_patterns(ctx, engine)
               frozen = RewritePatternSet.freeze(set)

               try do
                 time =
                   elapsed(fn ->
                     MLIR.Rewrite.apply_patterns!(ir, frozen, enable_folding: false)
                   end)

                 text = MLIR.to_string(ir)
                 ^count = length(Regex.scan(~r/llvm.call @enif_alloc/, text))
                 true = MLIR.verify?(ir)

                 Map.put(
                   time,
                   "callbacks",
                   if(pattern, do: MLIR.RewritePattern.callback_count(pattern), else: 0)
                 )
               after
                 if pdl_module, do: MLIR.Module.destroy(pdl_module)
                 MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen)
               end
             end)
           end)}
        end

      IO.puts(
        JSON.encode!(%{
          operations: count,
          repetitions: repeats,
          passes: passes,
          filtering: filters,
          allocation: alloc
        })
      )
    after
      Context.destroy(ctx)
    end
  end

  defp allocation_patterns(ctx, :pdl) do
    {set, pdl_module} =
      RewritePatternSet.with_pdl_patterns([Beaver.ENIF.UseENIFAlloc.replace_alloc()], ctx: ctx)

    {set, nil, pdl_module}
  end

  defp allocation_patterns(ctx, :beam) do
    set = RewritePatternSet.create(ctx)

    pattern =
      MLIR.RewritePattern.create("llvm.call",
        ctx: ctx,
        match_and_rewrite: fn _, op, rewriter, state ->
          if MLIR.Attribute.value(op[:callee]) == "malloc" do
            base = MLIR.PatternRewriter.as_base(rewriter)
            CAPI.mlirRewriterBaseStartOpModification(base, op)

            CAPI.mlirOperationSetAttributeByName(
              op,
              StringRef.create("callee"),
              MLIR.Attribute.flat_symbol_ref("enif_alloc", ctx: ctx)
            )

            CAPI.mlirRewriterBaseFinalizeOpModification(base, op)
            {:ok, state}
          else
            {:error, state}
          end
        end
      )

    RewritePatternSet.add(set, pattern)
    {set, pattern, nil}
  end

  defp with_module(ctx, source, fun) do
    ir = MLIR.Module.create!(source, ctx: ctx)

    try do
      fun.(ir)
    after
      MLIR.Module.destroy(ir)
    end
  end

  defp elapsed(fun) do
    {microseconds, _} = :timer.tc(fun)
    %{"microseconds" => microseconds}
  end

  defp measure(repeats, fun) do
    fun.()
    samples = for _ <- 1..repeats, do: fun.()
    median = samples |> Enum.sort_by(& &1["microseconds"]) |> Enum.at(div(repeats, 2))
    Map.put(median, "samples_us", Enum.map(samples, & &1["microseconds"]))
  end
end

args = System.argv()
count = args |> Enum.at(0, "1000") |> String.to_integer()
repetitions = args |> Enum.at(1, "5") |> String.to_integer()
if count < 1 or repetitions < 1, do: raise(ArgumentError, "counts must be positive")
CallbackBench.run(count, repetitions)
