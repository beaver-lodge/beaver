defmodule CUDARuntimeTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR

  @moduletag :cuda_runtime

  test "invoke main", %{ctx: ctx} do
    ctx = ctx
    MLIR.Context.register_translations(ctx)
    # trap sigchld when running ptxas to generate fatbin
    System.trap_signal(:sigchld, fn -> :ok end)

    libs = ~w{libmlir_cuda_runtime.so libmlir_runner_utils.so libmlir_c_runner_utils.so}

    jit =
      MLIR.Module.create(ctx, File.read!("test/gpu-to-cubin.mlir"))
      |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
      |> MLIR.Pass.Composer.append("gpu-lower-to-nvvm-pipeline{cubin-format=fatbin}")
      |> MLIR.Pass.Composer.run!()
      |> MLIR.ExecutionEngine.create!(
        shared_lib_paths: Enum.map(libs, &Path.join([:code.priv_dir(:beaver), "lib", &1]))
      )

    MLIR.ExecutionEngine.invoke!(jit, "main", [], nil, dirty: :cpu_bound)
    MLIR.ExecutionEngine.destroy(jit)
  end
end
