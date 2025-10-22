defmodule Beaver.ENIF.UseENIFAlloc do
  @moduledoc false
  use Beaver
  alias MLIR.Dialect.{LLVM, Func}
  use MLIR.Pass, on: Func.func()
  import Beaver.Pattern

  defmodule ReplaceLLVMOp do
    use Beaver.MLIR.RewritePattern

    def construct(nil) do
      {:ok, nil}
    end

    def destruct(state) do
      :ok
    end

    def match_and_rewrite(_pattern, op, rewriter, state) do
      if MLIR.Operation.name(op) == LLVM.call() do
        op[:callee] |> MLIR.Attribute.value() |> dbg
      end

      {:ok, state}
    end
  end

  def initialize(ctx, nil) do
    patterns = [
      replace_alloc(),
      replace_free()
    ]

    frozen_pat_set = Beaver.Pattern.compile_patterns(ctx, patterns)
    {:ok, %{patterns: frozen_pat_set, owned: true}}
  end

  def clone(%{patterns: frozen_pat_set}) do
    %{patterns: frozen_pat_set, owned: false}
  end

  def destruct(%{patterns: frozen_pat_set, owned: true}) do
    MLIR.CAPI.mlirFrozenRewritePatternSetDestroy(frozen_pat_set)
  end

  def destruct(%{owned: false}) do
    :ok
  end

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

  def run(op, %{patterns: patterns} = state) do
    MLIR.apply!(op, patterns)
    state
  end
end
