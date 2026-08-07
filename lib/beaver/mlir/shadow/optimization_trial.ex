defmodule Beaver.Shadow.OptimizationTrial do
  @moduledoc """
  A reproducible compiler-side optimization trial on real Triton IR.

  The trial answers one question: does a layout strategy reduce the number of
  `ttg.convert_layout` operations that survive in the TTGIR, and by how much?
  It is the first Shadow Wavefront consumer that compares two real pipeline
  variants on a real workload instead of a synthetic fixture.

  The baseline strategy runs `convert-triton-to-tritongpu` only. The optimized
  strategy additionally runs `tritongpu-remove-layout-conversions`, which
  propagates layouts through the graph and eliminates redundant conversions.
  Both variants are audited with `Beaver.MLIR.Triton.LayoutAudit`, so the
  reduction is a structural fact, not a subjective reading of the IR.
  """

  alias Beaver.MLIR

  defmodule Result do
    @moduledoc "The audited outcome of one trial."
    @enforce_keys [:input_digest, :baseline, :optimized, :reduced]
    defstruct [:input_digest, :baseline, :optimized, :reduced, :lowered_to_llvm]

    @type t() :: %__MODULE__{
            input_digest: String.t(),
            baseline: non_neg_integer(),
            optimized: non_neg_integer(),
            reduced: boolean(),
            lowered_to_llvm: boolean()
          }
  end

  @doc """
  Runs the layout trial on a real Triton IR module.

  Options:

    * `:target` — Triton GPU target, defaults to `cuda:80`
  """
  @spec run(MLIR.Module.t(), keyword()) :: Result.t()
  def run(%MLIR.Module{} = module, opts \\ []) do
    target = Keyword.get(opts, :target, "cuda:80")
    source_text = MLIR.to_string(module)
    context = MLIR.context(module)

    baseline =
      module
      |> Beaver.Composer.append("convert-triton-to-tritongpu{target=#{target}}")
      |> Beaver.Composer.run!()

    baseline_count = audit_count(baseline)

    optimized =
      baseline
      |> Beaver.Composer.append("tritongpu-remove-layout-conversions")
      |> Beaver.Composer.run!()

    optimized_count = audit_count(optimized)

    lowered_to_llvm =
      try do
        fresh = MLIR.Module.create!(source_text, ctx: context)

        # `compile_to_llvm` mutates and returns the same module (`fresh`), so
        # the text must be read before the module is destroyed.
        result =
          try do
            llvm = Beaver.Triton.compile_to_llvm(fresh, target: target)
            MLIR.to_string(llvm) =~ "llvm.func"
          after
            MLIR.Module.destroy(fresh)
          end

        result
      rescue
        exception ->
          IO.warn("compile_to_llvm failed: #{Exception.message(exception)}")
          false
      end

    %Result{
      input_digest:
        module
        |> MLIR.Bytecode.write!()
        |> then(&:crypto.hash(:sha256, &1))
        |> Base.encode16(case: :lower),
      baseline: baseline_count,
      optimized: optimized_count,
      reduced: optimized_count < baseline_count,
      lowered_to_llvm: lowered_to_llvm
    }
  end

  defp audit_count(module) do
    module
    |> MLIR.Triton.LayoutAudit.audit()
    |> Map.fetch!(:operation_count)
  end
end
