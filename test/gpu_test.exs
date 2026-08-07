defmodule GPUTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.GPU
  doctest Beaver.MLIR.Dialect.GPU

  @moduletag :cuda

  test "fatbin", %{ctx: ctx} do
    # trap sigchld when running ptxas to generate fatbin
    System.trap_signal(:sigchld, fn -> :ok end)

    module =
      File.read!("test/gpu-to-cubin.mlir")
      |> MLIR.Module.create!(ctx: ctx)

    target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx)

    assert module
           |> GPU.package_binary!(target, format: :fatbin)
           |> to_string() =~ "gpu.binary @other_func_kernel"
  end

  test "isa", %{ctx: ctx} do
    module =
      File.read!("test/gpu-to-cubin.mlir")
      |> MLIR.Module.create!(ctx: ctx)

    target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx)

    assert module
           |> GPU.package_binary!(target, format: :isa)
           |> to_string() =~ "gpu.binary @other_func_kernel"
  end
end
