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
    |> Beaver.Composer.append("convert-triton-to-tritongpu{target=cuda:90}")
    |> Beaver.Composer.append("convert-triton-gpu-to-llvm")
    |> Beaver.Composer.append("canonicalize")
    |> Beaver.Composer.run!()
    |> then(fn lowered ->
      text = MLIR.to_string(lowered)
      assert text =~ "llvm.func"
      lowered
    end)
  end

  @tag skip: !@enabled
  @tag :tmp_dir
  test "compiles a Triton kernel to PTX on the CPU", %{ctx: ctx, tmp_dir: tmp_dir} do
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

    lowered =
      module
      |> Beaver.Composer.append("convert-triton-to-tritongpu{target=cuda:90}")
      |> Beaver.Composer.append("convert-triton-gpu-to-llvm")
      |> Beaver.Composer.append("canonicalize")
      |> Beaver.Composer.run!()

    llvm_bin = System.get_env("LLVM_CONFIG_PATH") |> Path.dirname()
    mlir_path = Path.join(tmp_dir, "lowered.mlir")
    ll_path = Path.join(tmp_dir, "lowered.ll")
    ptx_path = Path.join(tmp_dir, "lowered.ptx")

    File.write!(mlir_path, MLIR.to_string(lowered))

    {_, 0} =
      System.cmd(
        Path.join(llvm_bin, "mlir-translate"),
        ["--mlir-to-llvmir", mlir_path, "-o", ll_path],
        stderr_to_stdout: true
      )

    {ptx_output, 0} =
      System.cmd(
        Path.join(llvm_bin, "llc"),
        ["-march=nvptx64", "-mcpu=sm_90", ll_path, "-o", ptx_path],
        stderr_to_stdout: true
      )

    ptx = File.read!(ptx_path)
    assert ptx =~ ".visible .entry addptr"
    assert ptx =~ ".target sm_90"
    assert ptx_output == ""
  end
end
