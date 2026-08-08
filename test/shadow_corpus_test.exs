defmodule Beaver.Shadow.CorpusTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias Beaver.MLIR.Triton.LayoutAudit
  alias Beaver.Shadow.Corpus
  alias Beaver.Shadow.OptimizationTrial

  @triton_enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  describe "fixture entries" do
    test "all fixtures are committed" do
      for fixture <- Corpus.fixtures() do
        assert File.exists?(Corpus.fixture_path(fixture.name)),
               "missing fixture #{fixture.file}"
      end
    end

    test "attention fixture contains the flash-attention pattern" do
      source = File.read!(Corpus.fixture_path(:attention))
      assert source =~ "tt.func public @attn_fwd"
      assert source =~ "tt.dot"
      assert source =~ "scf.for"
      assert source =~ "tt.reduce"
    end
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "TTIR fixtures reproduce the recorded baseline and optimized counts" do
    context = MLIR.Context.create(all_dialects: false)
    on_exit(fn -> MLIR.Context.destroy(context) end)
    Beaver.Triton.register(context)

    for fixture <- Corpus.fixtures(), fixture.dialect == :ttir do
      module = MLIR.Module.create!(File.read!(Corpus.fixture_path(fixture.name)), ctx: context)
      on_exit(fn -> MLIR.Module.destroy(module) end)

      result = OptimizationTrial.run(module)

      assert result.baseline == fixture.baseline, "baseline for #{fixture.name}"
      assert result.optimized == fixture.optimized, "optimized for #{fixture.name}"
      assert result.reduced
      assert result.lowered_to_llvm == fixture.lowered_to_llvm
    end
  end

  @tag :triton
  @tag skip: !@triton_enabled
  test "remat fixture audits to the recorded counts" do
    context = MLIR.Context.create(all_dialects: false)
    on_exit(fn -> MLIR.Context.destroy(context) end)
    Beaver.Triton.register(context)

    fixture = Corpus.fixture(:remat)
    module = MLIR.Module.create!(File.read!(Corpus.fixture_path(:remat)), ctx: context)
    on_exit(fn -> MLIR.Module.destroy(module) end)

    assert LayoutAudit.audit(module).operation_count == fixture.baseline

    optimized =
      module
      |> Beaver.Composer.append("tritongpu-remove-layout-conversions")
      |> Beaver.Composer.run!()

    assert LayoutAudit.audit(optimized).operation_count == fixture.optimized
  end
end
