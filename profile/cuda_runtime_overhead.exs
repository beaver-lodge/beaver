use Beaver
alias Beaver.MLIR

ctx = MLIR.Context.create()
MLIR.Context.register_translations(ctx)
System.trap_signal(:sigchld, fn -> :ok end)

libs = ~w{libmlir_cuda_runtime.so libmlir_runner_utils.so libmlir_c_runner_utils.so}

jit =
  MLIR.Module.create!(ctx, File.read!("test/gpu-to-cubin.mlir"))
  |> Beaver.Composer.nested("func.func", "llvm-request-c-wrappers")
  |> Beaver.Composer.append("gpu-lower-to-nvvm-pipeline{cubin-format=fatbin}")
  |> Beaver.Composer.run!()
  |> MLIR.ExecutionEngine.create!(
    shared_lib_paths: Enum.map(libs, &Path.join([:code.priv_dir(:beaver), "lib", &1]))
  )

MLIR.ExecutionEngine.invoke!(jit, "main", [], nil, dirty: :cpu_bound)
MLIR.ExecutionEngine.invoke!(jit, "main", [], nil, dirty: :io_bound)
MLIR.ExecutionEngine.invoke!(jit, "main", [], nil)
MLIR.ExecutionEngine.destroy(jit)
