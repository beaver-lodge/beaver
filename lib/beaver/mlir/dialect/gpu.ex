alias Beaver.MLIR.CAPI

defmodule Beaver.MLIR.Dialect.GPU do
  alias Beaver.Changeset
  alias Beaver.Deferred
  alias Beaver.MLIR
  alias Beaver.MLIR.{Attribute, Block, Operation, StringRef}
  alias Beaver.MLIR.Dialect

  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """

  use Beaver.MLIR.Dialect, dialect: "gpu", ops: Dialect.Registry.ops("gpu")

  @doc """
  Returns the name of the attribute containing the number of buffers located in the workgroup memory.
  ## Examples
    iex> Beaver.MLIR.Dialect.GPU.number_of_buffers_in_workgroup_attributions_attribute_name()
    :workgroup_attributions
  """
  def number_of_buffers_in_workgroup_attributions_attribute_name,
    do: CAPI.beaverGetNumWorkgroupAttributionsAttrName() |> to_string() |> String.to_atom()

  @doc """
  Get the name of the attribute used to annotate the modules that contain kernel modules.
  ## Examples
    iex> Beaver.MLIR.Dialect.GPU.container_module_attribute_name()
    :"gpu.container_module"
  """
  def container_module_attribute_name,
    do: CAPI.beaverGetContainerModuleAttrName() |> to_string() |> String.to_atom()

  @object_formats %{offload: 1, assembly: 2, binary: 3, fatbin: 4}

  @doc """
  Builds a `gpu.binary` module from an object value without printing or parsing
  MLIR text.

  `target` is an NVVM or ROCDL target attribute. `object` remains an Elixir
  binary all the way into the native `gpu.object` builder, so assembly and
  binary payloads do not require MLIR string escaping.

  The supported formats are `:offload`, `:assembly`, `:binary`, and `:fatbin`.
  """
  @spec binary_module(String.t(), Deferred.attribute(), binary(), keyword()) ::
          Deferred.contextual(MLIR.Module.t())
  def binary_module(name, target, object, opts \\ [])
      when is_binary(name) and is_binary(object) and is_list(opts) do
    format = opts |> Keyword.get(:format, :fatbin) |> object_format!()

    Deferred.from_opts(opts, fn ctx ->
      location = MLIR.Location.unknown(ctx: ctx)
      target = Deferred.resolve(target, ctx)

      object_attribute = CAPI.beaverGPUObjectAttrGet(target, format, StringRef.create(object))

      operation =
        %Changeset{name: "gpu.binary", context: ctx, location: location}
        |> Changeset.add_argument(sym_name: Attribute.string(name, ctx: ctx))
        |> Changeset.add_argument(objects: Attribute.array([object_attribute], ctx: ctx))
        |> Operation.create()

      module = MLIR.Module.empty(location)
      Block.append(MLIR.Module.body(module), operation)
      MLIR.verify!(module)
    end)
  end

  @doc "Returns the raw object payload stored in a `gpu.object` attribute."
  @spec object_data(MLIR.Attribute.t()) :: {:ok, binary()} | {:error, String.t()}
  def object_data(%MLIR.Attribute{} = attribute) do
    if CAPI.beaverAttributeIsAGPUObject(attribute) |> Beaver.Native.to_term(),
      do: {:ok, CAPI.beaverGPUObjectAttrGetObject(attribute) |> MLIR.to_string()},
      else: {:error, "attribute is not a gpu.object"}
  end

  @doc """
  Construct an NVVM target attribute from Elixir data.

  Supported options:

    * `:chip` - target chip, e.g. `"sm_80"`
    * `:triple` - target triple, e.g. `"nvptx64-nvidia-cuda"`
    * `:features` - target features, e.g. `"+ptx80"`
    * `:opt` - optimization level, an integer

  Returns a deferred attribute value unless `ctx:` is given, following the
  convention of `MLIR.Attribute.get/2`. The resulting attribute can be passed
  to `package_binary!/3` as the target configuration of a `gpu.binary`.
  """
  def nvvm_target_attribute(opts \\ []), do: target_attribute("nvvm", opts, [:chip, :triple, :features, :opt])

  @doc """
  Construct a ROCDL target attribute from Elixir data.

  Supported options:

    * `:chip` - target chip, e.g. `"gfx90a"` (defaults to `"gfx900"`)
    * `:triple` - target triple, e.g. `"amdgcn-amd-amdhsa"`
    * `:features` - target features, e.g. `"+wavefrontsize64"`
    * `:opt` - optimization level, an integer (defaults to `2`)
    * `:abi` - ABI version, e.g. `"600"`
    * `:arch` - processor, generic GPU, triple or full target ID; replaces `:chip`
    * `:wavesize` - `0` (target default), `32` or `64`, for lowering and codegen

  Returns a deferred attribute value unless `ctx:` is given, following the
  convention of `MLIR.Attribute.get/2`. The resulting attribute can be passed
  to `package_binary!/3` as the target configuration of a `gpu.binary`.
  """
  def rocdl_target_attribute(opts \\ []) do
    if Keyword.has_key?(opts, :arch) and Keyword.has_key?(opts, :chip),
      do: raise(ArgumentError, "use either :arch or :chip")

    wavesize = Keyword.get(opts, :wavesize, 0)
    unless wavesize in [0, 32, 64], do: raise(ArgumentError, "wavesize must be 0, 32 or 64")
    target = target_attribute("rocdl", opts, [:chip, :triple, :features, :opt, :abi])

    if Keyword.has_key?(opts, :arch) or Keyword.has_key?(opts, :wavesize) do
      Deferred.from_opts(opts, fn ctx ->
        unless rocdl_target_info_supported?(), do: raise(ArgumentError, "linked LLVM lacks ROCDL TargetInfo")
        arch = Keyword.get(opts, :arch, Keyword.get(opts, :chip, "gfx900"))

        configured =
          CAPI.beaverROCDLTargetAttrConfigure(Deferred.resolve(target, ctx), StringRef.create(arch), wavesize)

        if MLIR.null?(configured), do: raise(ArgumentError, "invalid or conflicting ROCDL arch, wavesize or features")
        configured
      end)
    else
      target
    end
  end

  def rocdl_target_info_supported?, do: CAPI.beaverROCDLTargetInfoSupported() |> Beaver.Native.to_term()

  defp target_attribute(dialect, opts, keys) do
    params =
      Enum.map_join(Keyword.take(opts, keys), ", ", fn
        {:opt, value} -> "O = #{value}"
        {name, value} when is_binary(value) -> ~s{#{name} = "#{value}"}
        {name, value} -> "#{name} = #{value}"
      end)

    MLIR.Attribute.get("##{dialect}.target<#{params}>", opts)
  end

  @doc """
  Lower and package a module containing GPU kernels into a `gpu.binary`.

  `targets` is an NVVM or ROCDL target attribute (from
  `nvvm_target_attribute/1` or `rocdl_target_attribute/1`) or a list of them;
  one `gpu.object` is produced per target. The conversion pass is selected
  from the target dialect: `#nvvm.target` lowers through `convert-gpu-to-nvvm`
  and `#rocdl.target` through `convert-gpu-to-rocdl`. The target configuration
  is attached to each lowered `gpu.module` as its `targets` attribute,
  replacing hard-coded legacy pipeline options.

  Options:

    * `:format` - binary format passed to `gpu-module-to-binary`. Defaults to
      `:isa` (PTX assembly), which is generated entirely on CPU without
      requiring a CUDA toolkit.

  Returns the module with `gpu.launch`/`gpu.func` code replaced by `gpu.binary`
  operations. Assembly (`:isa`) for both backends is generated entirely on CPU
  by the LLVM NVPTX/AMDGPU targets bundled with the MLIR prebuilt.
  """
  def package_binary!(module, targets, opts \\ []) do
    format = Keyword.get(opts, :format, :isa)
    ctx = MLIR.context(module)
    MLIR.Context.register_translations(ctx)

    targets = targets |> List.wrap() |> Enum.map(&Deferred.resolve(&1, ctx))
    if targets == [], do: raise(ArgumentError, "at least one target attribute is required")
    Enum.each(targets, &conversion_pass!/1)
    module = module |> Beaver.Composer.append("gpu-kernel-outlining") |> Beaver.Composer.run!()
    body = MLIR.Module.body(module)

    for gpu <- Enum.to_list(Beaver.Walker.operations(body)), Operation.name(gpu) == "gpu.module" do
      [first | rest] = targets
      binary = compile_target(gpu, first, format, ctx)

      try do
        Enum.each(rest, fn target ->
          next = compile_target(gpu, target, format, ctx)

          try do
            objects =
              Enum.to_list(binary["objects"]) ++ Enum.to_list(next["objects"])

            Operation.put_attribute(binary, "objects", Attribute.array(objects, ctx: ctx))
          after
            Operation.destroy(next)
          end
        end)

        CAPI.mlirBlockInsertOwnedOperationBefore(body, gpu, binary)
      rescue
        error ->
          Operation.destroy(binary)
          reraise error, __STACKTRACE__
      end

      Operation.destroy(gpu)
    end

    MLIR.verify!(module)
  end

  defp compile_target(gpu, target, format, ctx) do
    container = MLIR.Module.empty(MLIR.Location.unknown(ctx: ctx))
    clone = Operation.clone(gpu)
    Block.append(MLIR.Module.body(container), clone)

    try do
      container |> Beaver.Composer.nested("gpu.module", conversion_pass!(target)) |> Beaver.Composer.run!()
      if target_dialect(target) == "rocdl", do: CAPI.beaverROCDLTargetApplyModuleFlags(clone, target)
      Operation.put_attribute(clone, "targets", Attribute.array([target], ctx: ctx))
      container |> Beaver.Composer.append("gpu-module-to-binary{format=#{format}}") |> Beaver.Composer.run!()
      [binary] = Enum.to_list(Beaver.Walker.operations(MLIR.Module.body(container)))
      CAPI.mlirOperationRemoveFromParent(binary)
      binary
    after
      MLIR.Module.destroy(container)
    end
  end

  defp target_dialect(target),
    do: target |> CAPI.mlirAttributeGetDialect() |> CAPI.mlirDialectGetNamespace() |> MLIR.to_string()

  defp conversion_pass!(target) do
    case target_dialect(target) do
      "nvvm" ->
        "convert-gpu-to-nvvm"

      "rocdl" ->
        arch = CAPI.beaverROCDLTargetAttrGetArch(target) |> MLIR.to_string()
        wave = CAPI.beaverROCDLTargetAttrGetWaveSize(target) |> Beaver.Native.to_term()
        options = if rocdl_target_info_supported?(), do: "arch=#{arch} wavesize=#{wave}", else: "chipset=#{arch}"
        "convert-gpu-to-rocdl{#{options}}"

      other ->
        raise ArgumentError, "unsupported GPU target attribute dialect: #{other}"
    end
  end

  defp object_format!(format) when is_map_key(@object_formats, format), do: Map.fetch!(@object_formats, format)

  defp object_format!(format) do
    raise ArgumentError,
          "unsupported GPU object format #{inspect(format)}; expected one of #{inspect(Map.keys(@object_formats))}"
  end
end
