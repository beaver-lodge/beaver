defmodule GPUBinaryTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.GPU

  test "nvvm_target_attribute builds a target attribute from data", %{ctx: ctx} do
    assert MLIR.to_string(GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx)) ==
             ~s{#nvvm.target<chip = "sm_80">}

    full =
      GPU.nvvm_target_attribute(
        chip: "sm_80",
        triple: "nvptx64-nvidia-cuda",
        features: "+ptx80",
        opt: 3,
        ctx: ctx
      )
      |> MLIR.to_string()

    assert full =~ ~s{chip = "sm_80"}
    assert full =~ ~s{features = "+ptx80"}
    assert full =~ "O = 3"
  end

  test "package_binary! produces a gpu.binary with one object per target", %{ctx: ctx} do
    module =
      File.read!("test/gpu-to-cubin.mlir")
      |> MLIR.Module.create!(ctx: ctx)

    target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx)

    ir =
      module
      |> GPU.package_binary!(target, format: :isa)
      |> MLIR.to_string()

    assert ir =~ "gpu.binary @other_func_kernel"
    assert ir =~ ~s{#gpu.object<#nvvm.target<chip = "sm_80">}
    assert ir =~ ".entry other_func_kernel"
    refute ir =~ "gpu.module"
  end

  test "package_binary! packages multiple target objects into one binary", %{ctx: ctx} do
    module =
      File.read!("test/gpu-to-cubin.mlir")
      |> MLIR.Module.create!(ctx: ctx)

    targets = [
      GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx),
      GPU.nvvm_target_attribute(chip: "sm_90", ctx: ctx)
    ]

    ir =
      module
      |> GPU.package_binary!(targets, format: :isa)
      |> MLIR.to_string()

    assert ir =~ ~s{#gpu.object<#nvvm.target<chip = "sm_80">}
    assert ir =~ ~s{#gpu.object<#nvvm.target<chip = "sm_90">}
  end

  test "package_binary! accepts a standalone gpu.module", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        gpu.module @kernels {
          gpu.func @foo() {
            gpu.return
          }
        }
        """,
        ctx: ctx
      )

    target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx)

    ir =
      module
      |> GPU.package_binary!(target, format: :isa)
      |> MLIR.to_string()

    assert ir =~ "gpu.binary @kernels"
    assert ir =~ ~s{#gpu.object<#nvvm.target<chip = "sm_80">}
  end
end
