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

  test "rocdl_target_attribute builds a target attribute from data", %{ctx: ctx} do
    target = GPU.rocdl_target_attribute(chip: "gfx90a", ctx: ctx)
    assert MLIR.to_string(target) == ~s{#rocdl.target<chip = "gfx90a">}
    assert target |> MLIR.CAPI.beaverROCDLTargetAttrGetChip() |> MLIR.to_string() == "gfx90a"

    full =
      GPU.rocdl_target_attribute(
        chip: "gfx90a",
        triple: "amdgcn-amd-amdhsa-pal",
        features: "+wavefrontsize64",
        opt: 3,
        abi: "500",
        ctx: ctx
      )
      |> MLIR.to_string()

    assert full =~ ~s{chip = "gfx90a"}
    assert full =~ ~s{triple = "amdgcn-amd-amdhsa-pal"}
    assert full =~ ~s{features = "+wavefrontsize64"}
    assert full =~ "O = 3"
    assert full =~ ~s{abi = "500"}

    unless GPU.rocdl_target_info_supported?() do
      assert_raise ArgumentError, ~r/TargetInfo-based ROCDL lowering/, fn ->
        GPU.rocdl_target_attribute(arch: "gfx1100", wavesize: 32, ctx: ctx)
      end
    end
  end

  test "binary_module preserves object bytes without MLIR text escaping", %{ctx: ctx} do
    payload = <<"quote=\" slash=\\ newline=\n nul=", 0, " tail">>
    target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx)

    module = GPU.binary_module("kernel", target, payload, format: :assembly, ctx: ctx)

    binary =
      module
      |> MLIR.Module.body()
      |> Beaver.Walker.operations()
      |> Enum.find(&(MLIR.Operation.name(&1) == "gpu.binary"))

    assert {:ok, objects} = MLIR.Operation.fetch(binary, "objects")
    assert {:ok, object} = MLIR.Attribute.fetch(objects, 0)
    assert {:ok, ^payload} = GPU.object_data(object)
    assert MLIR.verify!(module) == module
  end

  @tag :rocdl_target_info
  test "ROCDL resolves full target IDs and validates wave sizes", %{ctx: ctx} do
    target =
      GPU.rocdl_target_attribute(
        arch: "amdgcn-amd-amdhsa--gfx942:xnack+:sramecc-",
        wavesize: 64,
        ctx: ctx
      )

    ir = MLIR.to_string(target)
    assert ir =~ ~s(chip = "gfx942")
    assert ir =~ "amdgpu9.42-amd-amdhsa"
    assert ir =~ "wavesize = 64"
    assert ir =~ "+wavefrontsize64"

    assert MLIR.CAPI.beaverROCDLTargetAttrGetArch(target) |> MLIR.to_string() ==
             "amdgcn-amd-amdhsa--gfx942:xnack+:sramecc-"

    for opts <- [
          [arch: "gfx942", wavesize: 32],
          [arch: "unknown"],
          [wavesize: 17],
          [arch: "gfx1100", wavesize: 32, features: "+wavefrontsize64"],
          [arch: "gfx942", chip: "gfx942"]
        ] do
      assert_raise ArgumentError, fn -> GPU.rocdl_target_attribute(opts ++ [ctx: ctx]) end
    end

    assert GPU.rocdl_target_attribute(arch: "gfx9-4-generic", ctx: ctx) |> MLIR.to_string() =~
             "gfx9-4-generic"

    assert GPU.rocdl_target_attribute(
             chip: "gfx942",
             triple: "amdgcn-amd-amdhsa-pal",
             wavesize: 64,
             ctx: ctx
           )
           |> MLIR.to_string() =~ "amdgpu9.42-amd-amdhsa-pal"
  end

  @tag :rocdl_target_info
  test "ROCDL compiles each target using its own wave size", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        gpu.module @kernels {
          gpu.func @empty() kernel {
            gpu.return
          }
        }
        """,
        ctx: ctx
      )

    targets =
      for wave <- [32, 64],
          do: GPU.rocdl_target_attribute(arch: "gfx1100", wavesize: wave, ctx: ctx)

    module = GPU.package_binary!(module, targets)
    [binary] = Enum.to_list(Beaver.Walker.operations(MLIR.Module.body(module)))

    [wave32, wave64] =
      Enum.map(binary["objects"], fn object ->
        {:ok, data} = GPU.object_data(object)
        data
      end)

    assert wave32 =~ ".amdhsa_wavefront_size32 1"
    assert wave64 =~ ".amdhsa_wavefront_size32 0"
  end

  @tag :rocdl_target_info
  test "ROCDL target-ID modifiers reach LLVM module flags and assembly", %{ctx: ctx} do
    module =
      MLIR.Module.create!(
        """
        gpu.module @kernels {
          gpu.func @empty() kernel { gpu.return }
        }
        """,
        ctx: ctx
      )

    target = GPU.rocdl_target_attribute(arch: "gfx942:xnack+:sramecc-", ctx: ctx)
    module = GPU.package_binary!(module, target)
    [binary] = Enum.to_list(Beaver.Walker.operations(MLIR.Module.body(module)))
    [object] = Enum.to_list(binary["objects"])
    assert {:ok, assembly} = GPU.object_data(object)
    assert assembly =~ "gfx942:sramecc-:xnack+"
  end

  test "binary_module rejects unknown object formats", %{ctx: ctx} do
    target = GPU.nvvm_target_attribute(chip: "sm_80", ctx: ctx)

    assert_raise ArgumentError, ~r/unsupported GPU object format/, fn ->
      GPU.binary_module("kernel", target, "ptx", format: :unknown, ctx: ctx)
    end
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

  test "package_binary! produces a rocdl gpu.binary with amdgcn assembly", %{ctx: ctx} do
    module =
      File.read!("test/gpu-to-cubin.mlir")
      |> MLIR.Module.create!(ctx: ctx)

    target = GPU.rocdl_target_attribute(chip: "gfx90a", ctx: ctx)

    ir =
      module
      |> GPU.package_binary!(target, format: :isa)
      |> MLIR.to_string()

    assert ir =~ "gpu.binary @other_func_kernel"
    assert ir =~ ~s{#gpu.object<#rocdl.target<chip = "gfx90a">}
    assert ir =~ "amdgcn_target"
    assert ir =~ "s_endpgm"
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

  test "package_binary! packages multiple rocdl targets into one binary", %{ctx: ctx} do
    module =
      File.read!("test/gpu-to-cubin.mlir")
      |> MLIR.Module.create!(ctx: ctx)

    targets = [
      GPU.rocdl_target_attribute(chip: "gfx90a", ctx: ctx),
      GPU.rocdl_target_attribute(chip: "gfx90c", ctx: ctx)
    ]

    ir =
      module
      |> GPU.package_binary!(targets, format: :isa)
      |> MLIR.to_string()

    assert ir =~ ~s{#gpu.object<#rocdl.target<chip = "gfx90a">}
    assert ir =~ ~s{#gpu.object<#rocdl.target<chip = "gfx90c">}
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

  test "package_binary! accepts a standalone rocdl gpu.module", %{ctx: ctx} do
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

    target = GPU.rocdl_target_attribute(chip: "gfx90a", ctx: ctx)

    ir =
      module
      |> GPU.package_binary!(target, format: :isa)
      |> MLIR.to_string()

    assert ir =~ "gpu.binary @kernels"
    assert ir =~ ~s{#gpu.object<#rocdl.target<chip = "gfx90a">}
  end

  test "package_binary! rejects unsupported target attributes", %{ctx: ctx} do
    module =
      File.read!("test/gpu-to-cubin.mlir")
      |> MLIR.Module.create!(ctx: ctx)

    target = MLIR.Attribute.integer(MLIR.Type.i32(ctx: ctx), 1)

    assert_raise ArgumentError, ~r/unsupported GPU target attribute dialect/, fn ->
      GPU.package_binary!(module, target)
    end
  end
end
