# Benchmarks three implementation mechanisms for replacing malloc/free with
# enif_alloc/enif_free (the Beaver.ENIF.UseENIFAlloc rewrite):
#
#   1. build-once pass - frozen RewritePatternSet built once (like the old
#      MLIR.Pass initialize), applied per llvm.func
#   2. plan           - reusable Beaver.MLIR.Conversion.Plan, materialized once
#      per module (with dynamic legality so the conversion driver fires the
#      patterns on malloc/free)
#   3. per-func apply - PDL pattern set rebuilt inside apply_patterns! for
#      every func (the "rebuild per call" variant)
#
# Inputs scale in both dimensions: number of functions and number of
# malloc/free pairs per function.
#
# Run: mix run bench/use_enif_alloc_bench.exs

defmodule EnifBenchPatterns do
  @moduledoc false
  use Beaver
  alias MLIR.Dialect.LLVM
  import Beaver.Pattern

  defpat replace_alloc(benefit: 10) do
    size = value()
    ptr_t = type()
    {op, _} = LLVM.call(size, callee: Attribute.flat_symbol_ref("malloc")) >>> {:op, [ptr_t]}

    rewrite op do
      r =
        LLVM.call(
          callee_operands: size,
          callee: Attribute.flat_symbol_ref("enif_alloc"),
          operand_segment_sizes: :infer,
          op_bundle_sizes: ~a{array<i32>}
        ) >>> ptr_t

      replace(op, with: r)
    end
  end

  defpat replace_free(benefit: 10) do
    ptr = value()
    {op, _} = LLVM.call(ptr, callee: Attribute.flat_symbol_ref("free")) >>> {:op, []}

    rewrite op do
      {enif_free, _} =
        LLVM.call(
          callee_operands: ptr,
          callee: Attribute.flat_symbol_ref("enif_free"),
          operand_segment_sizes: :infer,
          op_bundle_sizes: ~a{array<i32>}
        ) >>> {:op, []}

      replace(op, with: enif_free)
    end
  end
end

defmodule EnifBenchPlan do
  @moduledoc false
  use Beaver
  alias Beaver.MLIR
  alias MLIR.Conversion.Plan
  alias MLIR.Dialect.LLVM
  use Beaver.Pattern.Native

  def plan do
    Plan.new(mode: :partial)
    |> Plan.add_dynamically_legal_dialect("llvm", &dynamic_legal?/1, version: "1.0")
    |> Plan.add_pattern(replace_alloc(), version: "1.0")
    |> Plan.add_pattern(replace_free(), version: "1.0")
  end

  def dynamic_legal?(op) do
    case MLIR.Operation.fetch(op, "callee") do
      {:ok, callee} ->
        case MLIR.Attribute.value(callee) do
          name when name in ["malloc", "free"] -> :illegal
          _ -> :legal
        end

      :error ->
        :legal
    end
  end

  defrewrite replace_alloc(operation, rewriter, _state),
    root: LLVM.call(),
    operands: [size],
    results: [ptr],
    attributes: [{:callee, callee}] do
    if MLIR.Attribute.value(callee) == "malloc" do
      base = MLIR.PatternRewriter.as_base(rewriter)
      ctx = MLIR.context(operation)

      mlir ctx: ctx, ip: base do
        new_ptr =
          LLVM.call(
            callee_operands: size,
            callee: Attribute.flat_symbol_ref("enif_alloc"),
            operand_segment_sizes: :infer,
            op_bundle_sizes: ~a{array<i32>}
          ) >>> MLIR.Value.type(ptr)

        MLIR.RewriterBase.replace_op(base, operation, [new_ptr])
      end

      :ok
    else
      :error
    end
  end

  defrewrite replace_free(operation, rewriter, _state),
    root: LLVM.call(),
    operands: [ptr],
    results: [],
    attributes: [{:callee, callee}] do
    if MLIR.Attribute.value(callee) == "free" do
      base = MLIR.PatternRewriter.as_base(rewriter)
      ctx = MLIR.context(operation)

      mlir ctx: ctx, ip: base do
        LLVM.call(
          callee_operands: ptr,
          callee: Attribute.flat_symbol_ref("enif_free"),
          operand_segment_sizes: :infer,
          op_bundle_sizes: ~a{array<i32>}
        ) >>> []
      end

      MLIR.RewriterBase.replace_op(base, operation, [])
      :ok
    else
      :error
    end
  end
end

defmodule EnifBench do
  @moduledoc false
  alias Beaver.MLIR
  alias MLIR.Conversion.Plan

  def build_module(ctx, {n_funcs, pairs}) do
    funcs = for i <- 0..(n_funcs - 1), into: "", do: func_text(i, pairs)

    MLIR.Module.create!(
      "module { llvm.func @malloc(i64) -> !llvm.ptr \n llvm.func @free(!llvm.ptr) \n #{funcs} }",
      ctx: ctx
    )
  end

  defp func_text(i, pairs) do
    body =
      for p <- 1..pairs, into: "" do
        """
        %c#{p} = llvm.mlir.constant(#{p * 16} : i64) : i64
        %ptr#{p} = llvm.call @malloc(%c#{p}) : (i64) -> !llvm.ptr
        llvm.call @free(%ptr#{p}) : (!llvm.ptr) -> ()
        """
      end

    "llvm.func @f#{i}() { #{body} llvm.return }\n"
  end

  def funcs(module) do
    {_, ops} =
      module
      |> MLIR.Module.body()
      |> Beaver.Walker.prewalk([], fn
        %MLIR.Operation{} = op, acc ->
          if MLIR.Operation.name(op) == "llvm.func", do: {op, [op | acc]}, else: {op, acc}

        other, acc ->
          {other, acc}
      end)

    Enum.reverse(ops)
  end

  # 1. build-once: frozen set built once, applied per func
  def build_once_set(ctx) do
    {set, pdl_mod} =
      MLIR.RewritePatternSet.with_pdl_patterns(
        [EnifBenchPatterns.replace_alloc(), EnifBenchPatterns.replace_free()],
        ctx: ctx
      )

    frozen = MLIR.RewritePatternSet.freeze(set)
    {frozen, pdl_mod}
  end

  def build_once_apply({frozen, _pdl_mod}, module) do
    Enum.each(funcs(module), &MLIR.apply!(&1, frozen))
    module
  end

  # 2. plan: reusable plan, materialized once per module
  def plan_run(plan, module), do: Plan.run!(plan, module)

  # 3. per-func: PDL set rebuilt for every func
  def per_func(ctx, module) do
    Enum.each(funcs(module), fn func ->
      {set, pdl_mod} =
        MLIR.RewritePatternSet.with_pdl_patterns(
          [EnifBenchPatterns.replace_alloc(), EnifBenchPatterns.replace_free()],
          ctx: ctx
        )

      try do
        MLIR.Rewrite.apply_patterns!(func, set)
      after
        MLIR.Module.destroy(pdl_mod)
      end
    end)

    module
  end
end

alias Beaver.MLIR

ctx = MLIR.Context.create()
build_once = EnifBench.build_once_set(ctx)
plan = EnifBenchPlan.plan()

# warmup + sanity: every variant must actually rewrite malloc -> enif_alloc
for size <- [{2, 1}, {4, 2}] do
  refute_malloc = fn mod ->
    if String.contains?(MLIR.to_string(mod), "@malloc"),
      do: raise("variant left @malloc in place")
  end

  m = EnifBench.build_module(ctx, size)
  EnifBench.build_once_apply(build_once, m)
  refute_malloc.(m)
  MLIR.Module.destroy(m)

  m2 = EnifBench.build_module(ctx, size)
  EnifBench.plan_run(plan, m2)
  refute_malloc.(m2)
  MLIR.Module.destroy(m2)

  m3 = EnifBench.build_module(ctx, size)
  EnifBench.per_func(ctx, m3)
  refute_malloc.(m3)
  MLIR.Module.destroy(m3)
end

IO.puts("sanity check passed: all variants rewrite malloc/free")

Benchee.run(
  %{
    "build-once pass (frozen set once, per-func apply)" => fn module ->
      EnifBench.build_once_apply(build_once, module)
    end,
    "plan (materialize once per module)" => fn module ->
      EnifBench.plan_run(plan, module)
    end,
    "per-func apply_patterns (rebuild PDL per func)" => fn module ->
      EnifBench.per_func(ctx, module)
    end
  },
  inputs: %{
    "small: 8 funcs x 2 pairs" => {8, 2},
    "medium: 32 funcs x 8 pairs" => {32, 8},
    "large: 128 funcs x 32 pairs" => {128, 32},
    "xlarge: 512 funcs x 64 pairs" => {512, 64}
  },
  before_each: &EnifBench.build_module(ctx, &1),
  after_each: &MLIR.Module.destroy/1,
  time: 2,
  warmup: 1
)

MLIR.Module.destroy(elem(build_once, 1))
MLIR.Context.destroy(ctx)
