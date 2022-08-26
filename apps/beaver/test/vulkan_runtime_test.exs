defmodule VulkanRuntimeTest do
  use ExUnit.Case
  alias Beaver.MLIR

  @tag timeout: :infinity, vulkan: true
  @vulkan_ir File.read!("test/vulkan.mlir")
  test "run ir with vulkan" do
    import Beaver.MLIR.Sigils
    import MLIR.{Transforms, Conversion}

    jit =
      ~m"""
      #{@vulkan_ir}
      """
      |> canonicalize
      |> cse
      |> gpu_kernel_outlining
      |> MLIR.Pass.Composer.pipeline("fold-memref-subview-ops")
      |> convert_gpu_to_spirv
      |> MLIR.Pass.Composer.nested("spv.module", fn pm ->
        MLIR.Pass.pipeline!(pm, "spirv-lower-abi-attrs")
        MLIR.Pass.pipeline!(pm, "spirv-update-vce")
      end)
      |> convert_gpu_launch_to_vulkan_launch
      |> convert_memref_to_llvm
      |> MLIR.Pass.Composer.nested("func.func", fn pm ->
        MLIR.Pass.pipeline!(pm, "llvm-request-c-wrappers")
      end)
      |> convert_func_to_llvm
      |> reconcile_unrealized_casts
      |> launch_func_to_vulkan
      |> MLIR.Pass.Composer.run!(print: false)
      |> MLIR.ExecutionEngine.create!(
        shared_lib_paths: [
          Beaver.LLVM.Config.lib_dir() |> Path.join("libvulkan-runtime-wrappers.dylib"),
          Beaver.LLVM.Config.lib_dir() |> Path.join("libmlir_runner_utils.dylib")
        ]
      )

    for _ <- 1..2 do
      for _ <- 1..2 do
        Task.async(fn ->
          for _ <- 1..2 do
            MLIR.ExecutionEngine.invoke!(jit, "main", [])
          end
        end)
      end
      |> Task.await_many()
    end
  end
end
