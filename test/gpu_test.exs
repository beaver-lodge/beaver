defmodule GPUTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR

  @moduletag :cuda

  test "fatbin", %{ctx: ctx} do
    MLIR.Context.register_translations(ctx)
    # trap sigchld when running ptxas to generate fatbin
    System.trap_signal(:sigchld, fn -> :ok end)

    assert File.read!("test/gpu-to-cubin.mlir")
           |> MLIR.Module.create!(ctx: ctx)
           |> Beaver.Composer.append(
             "gpu-lower-to-nvvm-pipeline{cubin-format=fatbin cubin-chip=sm_80}"
           )
           |> Beaver.Composer.run!()
           |> to_string() =~ "gpu.binary @other_func_kernel"
  end

  test "isa", %{ctx: ctx} do
    MLIR.Context.register_translations(ctx)

    assert MLIR.Module.create!(File.read!("test/gpu-to-cubin.mlir"), ctx: ctx)
           |> Beaver.Composer.append(
             "gpu-lower-to-nvvm-pipeline{cubin-format=isa cubin-chip=sm_80}"
           )
           |> Beaver.Composer.run!()
           |> to_string() =~ "gpu.binary @other_func_kernel"
  end
end
