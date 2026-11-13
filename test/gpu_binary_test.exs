defmodule GPUBinaryTest do
  use Beaver.Case, async: true
  alias Beaver.{MLIR, Walker}
  alias Beaver.MLIR.{Attribute, CAPI, Module, Operation, Type}
  alias Beaver.MLIR.Dialect.GPU

  @targets [
    {:nvvm, "sm_80", "sm_90", "nvptx64-nvidia-cuda", "+ptx80", []},
    {:rocdl, "gfx90a", "gfx90c", "amdgcn-amd-amdhsa-pal", "+wavefrontsize64", [abi: "500"]}
  ]
  for {backend, chip, second, triple, features, extra} <- @targets do
    @tag target: {backend, chip, triple, features, extra}
    test "#{backend} target attributes preserve target configuration", %{ctx: ctx, target: spec} do
      {backend, chip, triple, features, extra} = spec
      attr = target(backend, chip, ctx)
      assert MLIR.to_string(attr) == ~s(##{backend}.target<chip = "#{chip}">)
      if backend == :rocdl, do: assert(to_string(CAPI.beaverROCDLTargetAttrGetChip(attr)) == chip)
      opts = [chip: chip, triple: triple, features: features, opt: 3, ctx: ctx] ++ extra
      full = apply(GPU, :"#{backend}_target_attribute", [opts]) |> MLIR.to_string()

      for part <- [~s(chip = "#{chip}"), ~s(features = "#{features}"), "O = 3"],
          do: assert(full =~ part)

      if backend == :rocdl do
        assert full =~ ~s(triple = "#{triple}")
        assert full =~ ~s(abi = "500")
      end
    end

    for kind <- [:single, :multiple, :standalone] do
      @tag packaging: {backend, chip, second, kind}
      test "#{backend} packages #{kind} GPU modules", %{ctx: ctx, packaging: spec} do
        {backend, chip, second, kind} = spec
        source = if kind == :standalone, do: standalone("foo", ""), else: File.read!("test/gpu-to-cubin.mlir")
        chips = if kind == :multiple, do: [chip, second], else: [chip]
        targets = Enum.map(chips, &target(backend, &1, ctx))
        targets = if kind == :multiple, do: targets, else: hd(targets)

        Module.with_module(ctx, source, fn module ->
          ir = GPU.package_binary!(module, targets, format: :isa) |> MLIR.to_string()

          for chip <- chips,
              do: assert(ir =~ ~s(#gpu.object<##{backend}.target<chip = "#{chip}">))

          assert ir =~
                   "gpu.binary @#{if kind == :standalone, do: "kernels", else: "other_func_kernel"}"

          refute ir =~ "gpu.module"

          if kind == :single do
            for part <- if(backend == :nvvm, do: [".entry other_func_kernel"], else: ["amdgcn_target", "s_endpgm"]),
                do: assert(ir =~ part)
          end
        end)
      end
    end
  end

  test "binary_module preserves arbitrary object bytes", %{ctx: ctx} do
    payload = <<"quote=\" slash=\\ newline=\n nul=", 0, " tail">>

    Module.with_module(
      GPU.binary_module("kernel", target(:nvvm, "sm_80", ctx), payload, format: :assembly, ctx: ctx),
      fn module ->
        [binary] = Walker.find_ops(module, "gpu.binary")
        assert {:ok, objects} = Operation.fetch(binary, "objects")
        assert {:ok, object} = Attribute.fetch(objects, 0)
        assert {:ok, ^payload} = GPU.object_data(object)
        assert MLIR.verify!(module) == module
      end
    )
  end

  @tag :rocdl_target_info
  test "ROCDL resolves full target IDs and validates wave sizes", %{ctx: ctx} do
    arch = "amdgcn-amd-amdhsa--gfx942:xnack+:sramecc-"
    target = GPU.rocdl_target_attribute(arch: arch, wavesize: 64, ctx: ctx)
    ir = MLIR.to_string(target)

    for part <- [~s(chip = "gfx942"), "amdgpu9.42-amd-amdhsa", "wavesize = 64", "+wavefrontsize64"],
        do: assert(ir =~ part)

    assert to_string(CAPI.beaverROCDLTargetAttrGetArch(target)) == arch

    for opts <- [
          [arch: "gfx942", wavesize: 32],
          [arch: "unknown"],
          [wavesize: 17],
          [arch: "gfx1100", wavesize: 32, features: "+wavefrontsize64"],
          [arch: "gfx942", chip: "gfx942"]
        ],
        do: assert_raise(ArgumentError, fn -> GPU.rocdl_target_attribute(opts ++ [ctx: ctx]) end)

    for {opts, expected} <- [
          {[arch: "gfx9-4-generic"], "gfx9-4-generic"},
          {[chip: "gfx942", triple: "amdgcn-amd-amdhsa-pal", wavesize: 64], "amdgpu9.42-amd-amdhsa-pal"}
        ],
        do: assert(MLIR.to_string(GPU.rocdl_target_attribute(opts ++ [ctx: ctx])) =~ expected)
  end

  @tag :rocdl_target_info
  test "ROCDL compiles each target using its own wave size", %{ctx: ctx} do
    targets = for wave <- [32, 64], do: GPU.rocdl_target_attribute(arch: "gfx1100", wavesize: wave, ctx: ctx)
    [wave32, wave64] = package_objects(ctx, targets)
    assert wave32 =~ ".amdhsa_wavefront_size32 1"
    assert wave64 =~ ".amdhsa_wavefront_size32 0"
  end

  @tag :rocdl_target_info
  test "ROCDL target-ID modifiers reach LLVM module flags and assembly", %{ctx: ctx} do
    [assembly] = package_objects(ctx, GPU.rocdl_target_attribute(arch: "gfx942:xnack+:sramecc-", ctx: ctx))
    assert assembly =~ "gfx942:sramecc-:xnack+"
  end

  test "binary_module rejects unknown object formats", %{ctx: ctx} do
    assert_raise ArgumentError, ~r/unsupported GPU object format/, fn ->
      GPU.binary_module("kernel", target(:nvvm, "sm_80", ctx), "ptx", format: :unknown, ctx: ctx)
    end
  end

  test "package_binary! rejects unsupported target attributes", %{ctx: ctx} do
    Module.with_module(ctx, File.read!("test/gpu-to-cubin.mlir"), fn module ->
      assert_raise ArgumentError, ~r/unsupported GPU target attribute dialect/, fn ->
        GPU.package_binary!(module, Attribute.integer(Type.i32(ctx: ctx), 1))
      end
    end)
  end

  defp target(backend, chip, ctx),
    do: apply(GPU, :"#{backend}_target_attribute", [[chip: chip, ctx: ctx]])

  defp standalone(name, attrs),
    do: "gpu.module @kernels { gpu.func @#{name}() #{attrs} { gpu.return } }"

  defp package_objects(ctx, targets) do
    Module.with_module(ctx, standalone("empty", "kernel"), fn module ->
      module = GPU.package_binary!(module, targets)
      [binary] = Enum.to_list(Walker.operations(Module.body(module)))

      Enum.map(binary["objects"], fn object ->
        assert {:ok, data} = GPU.object_data(object)
        data
      end)
    end)
  end
end
