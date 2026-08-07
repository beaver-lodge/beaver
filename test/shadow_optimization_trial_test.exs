defmodule Beaver.Shadow.OptimizationTrialTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias Beaver.Shadow.OptimizationTrial

  @fixture_path Path.expand("fixtures/triton/ttgir_matmul.mlir", __DIR__)
  @triton_enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  @tag :triton
  @tag skip: !@triton_enabled
  test "real matmul TTGIR shows a reproducible convert_layout reduction" do
    context = MLIR.Context.create(all_dialects: false)
    on_exit(fn -> MLIR.Context.destroy(context) end)
    Beaver.Triton.register(context)

    module = MLIR.Module.create!(File.read!(@fixture_path), ctx: context)
    on_exit(fn -> MLIR.Module.destroy(module) end)

    result = OptimizationTrial.run(module)

    assert result.reduced
    assert result.baseline > result.optimized
    assert is_binary(result.input_digest) and byte_size(result.input_digest) == 64
  end
end
