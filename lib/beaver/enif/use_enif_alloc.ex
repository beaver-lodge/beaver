defmodule Beaver.ENIF.UseENIFAlloc do
  @moduledoc """
  Replace LLVM call ops of `malloc` and `free` with `enif_alloc` and `enif_free`.
  """
  use Beaver
  alias MLIR.Dialect.LLVM
  use MLIR.Pass, on: LLVM.func()
  import Beaver.Pattern

  defmodule ReplaceLLVMOp do
    @moduledoc false
    use Beaver.MLIR.RewritePattern

    def construct(nil) do
      {:ok, :llvm_pat_state}
    end

    def destruct(_state) do
      :ok
    end

    def match_and_rewrite(_pattern, op, rewriter, state) do
      if MLIR.Operation.name(op) == LLVM.call() and MLIR.Attribute.value(op[:callee]) == "malloc" do
        ctx = MLIR.context(op)
        ptr = MLIR.Operation.result(op, 0)

        mlir ctx: ctx, ip: rewriter do
          new_ptr =
            LLVM.call(
              callee_operands: Beaver.Walker.operands(op) |> Enum.to_list(),
              callee: Attribute.flat_symbol_ref("enif_alloc"),
              operand_segment_sizes: :infer,
              op_bundle_sizes: ~a{array<i32>}
            ) >>> MLIR.Value.type(ptr)

          MLIR.PatternRewriter.replace(rewriter, ptr, new_ptr)
          MLIR.PatternRewriter.erase_op(rewriter, op)
        end

        {:ok, state}
      else
        {:error, state}
      end
    end
  end

  def initialize(ctx, nil) do
    patterns = [
      replace_alloc(),
      replace_free()
    ]

    set = Beaver.Pattern.compile_patterns(ctx, patterns)

    frozen_set =
      set
      |> MLIR.RewritePatternSet.add(LLVM.call(), ReplaceLLVMOp, ctx: ctx)
      |> MLIR.RewritePatternSet.freeze()

    {:ok, %{patterns: frozen_set, owning: true, ctx: ctx}}
  end

  def clone(%{patterns: frozen_set, owning: true}) do
    %{patterns: frozen_set, owning: false}
  end

  def destruct(%{patterns: frozen_set, owning: true, ctx: ctx}) do
    MLIR.FrozenRewritePatternSet.threaded_destroy(ctx, frozen_set)
  end

  def destruct(%{owning: false}) do
    :ok
  end

  # the state is nil when initialization fails
  def destruct(nil) do
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

  def run(op, %{patterns: patterns, owning: false} = state) do
    MLIR.apply!(op, patterns)
    state
  end
end
