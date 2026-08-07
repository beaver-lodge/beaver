defmodule CUDARuntimeTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.CUDA
  alias Beaver.MLIR.Dialect.GPU

  @moduletag :cuda_runtime

  test "launch a kernel from gpu.binary with the Zig runner", %{ctx: ctx} do
    if CUDA.available?() do
      module =
        File.read!("test/gpu-to-cubin.mlir")
        |> MLIR.Module.create!(ctx: ctx)

      target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx)

      {:ok, cuda_module} =
        module
        |> GPU.package_binary!(target, format: :isa)
        |> CUDA.load_gpu_binary()

      {:ok, function} = CUDA.module_get_function(cuda_module, "other_func_kernel")

      n = 32
      {:ok, device_ptr} = CUDA.mem_alloc(n * 4)

      scalar = 42.0

      :ok =
        CUDA.launch_kernel(
          function,
          {1, 1, 1},
          {n, 1, 1},
          [
            {:f32, scalar},
            {:ptr, device_ptr},
            {:ptr, device_ptr},
            {:i64, 0},
            {:i64, n},
            {:i64, 1}
          ]
        )

      {:ok, data} = CUDA.memcpy_dtoh(device_ptr, n * 4)

      floats =
        for <<f::float-32-little <- data>> do
          f
        end

      assert Enum.all?(floats, &(&1 == scalar))

      :ok = CUDA.mem_free(device_ptr)
      :ok = CUDA.module_unload(cuda_module)
    else
      # No CUDA driver: the runner must degrade instead of crashing.
      assert {:error, reason} = CUDA.device_count()
      assert is_binary(reason)
    end
  end
end
