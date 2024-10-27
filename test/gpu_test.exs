defmodule GPUTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR

  @moduletag :smoke
  @moduletag :cuda

  test "fatbin", %{ctx: ctx} do
    ctx = ctx
    MLIR.Context.register_translations(ctx)
    # trap sigchld when running ptxas to generate fatbin
    System.trap_signal(:sigchld, fn -> :ok end)

    assert MLIR.Module.create(ctx, File.read!("test/gpu-to-cubin.mlir"))
           |> MLIR.Pass.Composer.append("gpu-lower-to-nvvm-pipeline{cubin-format=fatbin}")
           |> MLIR.Pass.Composer.run!()
           |> to_string() =~ "gpu.binary @other_func_kernel"
  end

  test "isa", %{ctx: ctx} do
    ctx = ctx
    MLIR.Context.register_translations(ctx)

    assert MLIR.Module.create(ctx, File.read!("test/gpu-to-cubin.mlir"))
           |> MLIR.Pass.Composer.append("gpu-lower-to-nvvm-pipeline{cubin-format=isa}")
           |> MLIR.Pass.Composer.run!()
           |> to_string() =~ "gpu.binary @other_func_kernel"
  end
end
