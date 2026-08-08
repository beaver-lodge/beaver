defmodule Beaver.Shadow.Corpus do
  @moduledoc """
  The TTGIR/TTIR corpus that drives Shadow Wavefront measurements.

  Each entry pins a fixture file plus the structural facts observed on the
  pinned Triton prebuilt (LLVM revision recorded by
  `Beaver.MLIR.Experiment` receipts at run time). `baseline`/`optimized` are
  the `ttg.convert_layout` counts before and after
  `tritongpu-remove-layout-conversions`; asserting them makes the CPU-only
  baseline receipts reproducible across runs and revisions.

  `:matmul` and `:attention` are TTIR kernels that lower through the full
  NVIDIA pipeline. `:remat` is an already-lowered TTGIR slice (a backward
  pass with `scf.for` + remat-style `ttg.convert_layout`s) used as the
  regression probe for triton-lang/triton#11026; it is audited in place
  because the pinned prebuilt cannot lower it to LLVM (see the probe task).
  """

  @fixtures [
    %{
      name: :matmul,
      file: "ttgir_matmul.mlir",
      dialect: :ttir,
      baseline: 24,
      optimized: 5,
      lowered_to_llvm: true
    },
    %{
      name: :attention,
      file: "ttir_attention.mlir",
      dialect: :ttir,
      baseline: 34,
      optimized: 5,
      lowered_to_llvm: true
    },
    %{
      name: :remat,
      file: "ttgir_convert_layout.mlir",
      dialect: :ttgir,
      baseline: 4,
      optimized: 1,
      lowered_to_llvm: false
    }
  ]

  @type fixture() :: %{
          name: atom(),
          file: String.t(),
          dialect: :ttir | :ttgir,
          baseline: non_neg_integer(),
          optimized: non_neg_integer(),
          lowered_to_llvm: boolean()
        }

  @spec fixtures() :: [fixture()]
  def fixtures, do: @fixtures

  @spec fixture(atom()) :: fixture()
  def fixture(name) when is_atom(name) do
    Enum.find(@fixtures, &(&1.name == name)) ||
      raise ArgumentError, "unknown corpus fixture: #{inspect(name)}"
  end

  @spec fixture_path(atom()) :: Path.t()
  def fixture_path(name) do
    Path.expand("../../../test/fixtures/triton/#{fixture(name).file}", __DIR__)
  end
end
