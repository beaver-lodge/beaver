defmodule Beaver.ENIF.UseENIFAlloc do
  @moduledoc "Replaces LLVM calls to malloc/free with the corresponding ENIF allocator."
  alias Beaver.MLIR
  alias MLIR.{Attribute, Operation, PatternRewriter, RewritePatternSet, RewriterBase}
  use MLIR.Pass, on: "llvm.func"

  def initialize(ctx, nil) do
    patterns =
      RewritePatternSet.create(ctx)
      |> RewritePatternSet.add("llvm.call", &replace_allocator/4, ctx: ctx)
      |> RewritePatternSet.freeze()

    {:ok, %{patterns: patterns, owning: true, ctx: ctx}}
  end

  def replace_allocator(_pattern, op, rewriter, state) do
    callee = op[:callee]
    name = if callee, do: Attribute.value(callee)

    if replacement = %{"malloc" => "enif_alloc", "free" => "enif_free"}[name] do
      base = PatternRewriter.as_base(rewriter)
      RewriterBase.start_op_modification(base, op)

      Operation.get_and_update(op, :callee, fn _ ->
        {nil, Attribute.flat_symbol_ref(replacement, ctx: MLIR.context(op))}
      end)

      RewriterBase.finalize_op_modification(base, op)
      {:ok, state}
    else
      {:error, state}
    end
  end

  def clone(%{patterns: patterns}), do: %{patterns: patterns, owning: false}

  def destruct(%{patterns: patterns, owning: true, ctx: ctx}),
    do: MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, patterns)

  def destruct(_state), do: :ok

  def run(op, %{patterns: patterns} = state) do
    MLIR.Rewrite.apply_patterns!(op, patterns)
    state
  end
end
