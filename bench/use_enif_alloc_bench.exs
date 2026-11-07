# mix run bench/use_enif_alloc_bench.exs
# BENCH_INPUTS=small|small_medium|large|all; BENCH_TIME=2; BENCH_WARMUP=1.
# Compare reused PDL, a reusable conversion plan, and PDL rebuilt per function.
defmodule EnifBenchPlan do
  use Beaver
  alias MLIR.Conversion.Plan
  alias MLIR.Dialect.LLVM
  use Beaver.Pattern.Native

  def plan do
    Plan.new(mode: :partial)
    |> Plan.add_dynamically_legal_dialect("llvm", &is_nil(target(&1)), version: "1.0")
    |> Plan.add_pattern(replace_call(), version: "1.0")
  end

  defp target(op) do
    case op[:callee] do
      nil -> nil
      attr -> %{"malloc" => "enif_alloc", "free" => "enif_free"}[MLIR.Attribute.value(attr)]
    end
  end

  defrewrite replace_call(operation, rewriter, _state), root: LLVM.call(), operands: [arg] do
    if name = target(operation) do
      base = MLIR.PatternRewriter.as_base(rewriter)
      types = Beaver.Walker.results(operation) |> Enum.map(&MLIR.Value.type/1)

      mlir ctx: MLIR.context(operation), ip: base do
        {_, results} =
          LLVM.call(
            callee_operands: arg,
            callee: Attribute.flat_symbol_ref(name),
            operand_segment_sizes: :infer,
            op_bundle_sizes: ~a{array<i32>}
          ) >>> {:op, types}

        MLIR.RewriterBase.replace_op(base, operation, results)
      end

      :ok
    else
      :error
    end
  end
end

defmodule EnifBench do
  alias Beaver.{MLIR, Native, Walker}
  alias Beaver.ENIF.UseENIFAlloc

  defp build(ctx, {n_funcs, pairs}) do
    body =
      Enum.map_join(1..pairs, fn n ->
        "%c#{n} = llvm.mlir.constant(#{n * 16} : i64) : i64 " <>
          "%p#{n} = llvm.call @malloc(%c#{n}) : (i64) -> !llvm.ptr " <>
          "llvm.call @free(%p#{n}) : (!llvm.ptr) -> () "
      end)

    funcs = Enum.map_join(1..n_funcs, &"llvm.func @f#{&1}() { #{body} llvm.return }")

    declarations =
      "llvm.func @malloc(i64) -> !llvm.ptr llvm.func @free(!llvm.ptr) " <>
        "llvm.func @enif_alloc(i64) -> !llvm.ptr llvm.func @enif_free(!llvm.ptr)"

    MLIR.Module.create!("module { #{declarations} #{funcs} }", ctx: ctx)
  end

  defp with_patterns(ctx, fun) do
    {:ok, state} = UseENIFAlloc.initialize(ctx, nil)
    Native.with_resource(state, &UseENIFAlloc.destruct/1, fun)
  end

  defp each_function(ir, fun) do
    ir |> MLIR.Module.body() |> Walker.operations() |> Enum.each(fun)
    ir
  end

  defp validate(ir) do
    if Regex.match?(~r/llvm.call @(malloc|free)\(/, MLIR.to_string(ir)),
      do: raise("variant left malloc/free calls in place")

    MLIR.verify!(ir)
  end

  defp seconds(name, default), do: System.get_env(name, default) |> Float.parse() |> elem(0)

  def run do
    Native.with_resource(MLIR.Context.create(), &MLIR.Context.destroy/1, fn ctx ->
      with_patterns(ctx, fn state ->
        plan = EnifBenchPlan.plan()

        jobs = %{
          "reused PDL / per-function apply" =>
            &each_function(&1, fn op -> UseENIFAlloc.run(op, state) end),
          "plan / materialize per module" => &MLIR.Conversion.Plan.run!(plan, &1),
          "PDL / rebuild per function" =>
            &each_function(&1, fn op ->
              with_patterns(ctx, fn fresh -> UseENIFAlloc.run(op, fresh) end)
            end)
        }

        for size <- [{2, 1}, {4, 2}], {_, job} <- jobs do
          Native.with_resource(build(ctx, size), &MLIR.Module.destroy/1, fn ir ->
            job.(ir) |> validate()
          end)
        end

        sizes =
          case System.get_env("BENCH_INPUTS", "all") do
            "small" -> [{8, 2}]
            "small_medium" -> [{8, 2}, {32, 8}]
            "large" -> [{128, 32}]
            _ -> [{8, 2}, {32, 8}, {128, 32}, {512, 64}]
          end

        Benchee.run(jobs,
          inputs:
            Map.new(sizes, fn {funcs, pairs} = size ->
              {"#{funcs} funcs x #{pairs} pairs", size}
            end),
          before_each: &build(ctx, &1),
          after_each: fn ir -> Native.with_resource(ir, &MLIR.Module.destroy/1, &validate/1) end,
          time: seconds("BENCH_TIME", "2"),
          warmup: seconds("BENCH_WARMUP", "1")
        )
      end)
    end)
  end
end

EnifBench.run()
