defmodule TritonTest do
  use Beaver.Case, async: true

  alias Beaver.MLIR

  @moduletag :triton

  @enabled System.get_env("BEAVER_TRITON_PREBUILT_DIR") != nil

  setup_all do
    if @enabled do
      Beaver.Triton.register_passes()
    end

    :ok
  end

  @tag skip: !@enabled
  test "registers Triton dialects and roundtrips ttir" do
    ctx = MLIR.Context.create(all_dialects: false)
    Beaver.Triton.register(ctx)

    module =
      MLIR.Module.create!(
        ~S"""
        module {
          tt.func @load_store(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %mask : i1) {
            %a = tt.load %ptr : !tt.ptr<f32>
            %b = tt.load %ptr, %mask : !tt.ptr<f32>
            tt.store %ptr, %a : !tt.ptr<f32>
            tt.store %ptr, %b, %mask : !tt.ptr<f32>
            tt.return
          }
        }
        """,
        ctx: ctx
      )

    text = MLIR.to_string(module, generic: false)
    assert text =~ "tt.func @load_store"
    assert text =~ "tt.load"
    assert text =~ "tt.store"
    MLIR.Context.destroy(ctx)
  end

  @tag skip: !@enabled
  test "runs a Triton ttgir-to-LLVM pipeline on the CPU" do
    ctx = MLIR.Context.create(all_dialects: false)
    Beaver.Triton.register(ctx)

    module =
      MLIR.Module.create!(
        ~S"""
        module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
          tt.func @addptr(%ptr: !tt.ptr<f32>, %i: i32) {
            %0 = tt.addptr %ptr, %i : !tt.ptr<f32>, i32
            tt.return
          }
        }
        """,
        ctx: ctx
      )

    module
    |> Beaver.Composer.append("convert-triton-to-tritongpu")
    |> Beaver.Composer.append("convert-tritongpu-to-llvm")
    |> Beaver.Composer.run!()
    |> then(fn lowered ->
      text = MLIR.to_string(lowered)
      assert text =~ "llvm.func"
      lowered
    end)

    MLIR.Context.destroy(ctx)
  end
end
