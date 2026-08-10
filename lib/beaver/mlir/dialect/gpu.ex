defmodule Beaver.MLIR.Dialect.GPU do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect

  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """

  use Beaver.MLIR.Dialect,
    dialect: "gpu",
    ops: Dialect.Registry.ops("gpu")

  @doc """
  Returns the name of the attribute containing the number of buffers located in the workgroup memory.
  ## Examples
    iex> Beaver.MLIR.Dialect.GPU.number_of_buffers_in_workgroup_attributions_attribute_name()
    :workgroup_attributions
  """
  def number_of_buffers_in_workgroup_attributions_attribute_name do
    MLIR.CAPI.beaverGetNumWorkgroupAttributionsAttrName() |> to_string() |> String.to_atom()
  end

  @doc """
  Get the name of the attribute used to annotate the modules that contain kernel modules.
  ## Examples
    iex> Beaver.MLIR.Dialect.GPU.container_module_attribute_name()
    :"gpu.container_module"
  """
  def container_module_attribute_name do
    MLIR.CAPI.beaverGetContainerModuleAttrName() |> to_string() |> String.to_atom()
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
  def nvvm_target_attribute(opts \\ []) do
    params =
      opts
      |> Keyword.take([:chip, :triple, :features, :opt])
      |> Enum.map(fn
        {:opt, value} -> "O = #{value}"
        {name, value} when is_binary(value) -> ~s{#{name} = "#{value}"}
        {name, value} -> "#{name} = #{value}"
      end)

    MLIR.Attribute.get("#nvvm.target<#{Enum.join(params, ", ")}>", opts)
  end

  @doc """
  Construct a ROCDL target attribute from Elixir data.

  Supported options:

    * `:chip` - target chip, e.g. `"gfx90a"` (defaults to `"gfx900"`)
    * `:triple` - target triple, e.g. `"amdgcn-amd-amdhsa"`
    * `:features` - target features, e.g. `"+wavefrontsize64"`
    * `:opt` - optimization level, an integer (defaults to `2`)
    * `:abi` - ABI version, e.g. `"600"`

  Returns a deferred attribute value unless `ctx:` is given, following the
  convention of `MLIR.Attribute.get/2`. The resulting attribute can be passed
  to `package_binary!/3` as the target configuration of a `gpu.binary`.
  """
  def rocdl_target_attribute(opts \\ []) do
    params =
      opts
      |> Keyword.take([:chip, :triple, :features, :opt, :abi])
      |> Enum.map(fn
        {:opt, value} -> "O = #{value}"
        {name, value} when is_binary(value) -> ~s{#{name} = "#{value}"}
        {name, value} -> "#{name} = #{value}"
      end)

    MLIR.Attribute.get("#rocdl.target<#{Enum.join(params, ", ")}>", opts)
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

    targets = List.wrap(targets)
    conversion = conversion_pass!(targets)

    lowered =
      module
      |> Beaver.Composer.append("gpu-kernel-outlining")
      |> Beaver.Composer.nested("gpu.module", conversion)
      |> Beaver.Composer.run!()

    targets_attr = MLIR.Attribute.array(targets, ctx: ctx)

    lowered
    |> MLIR.Module.body()
    |> Beaver.Walker.operations()
    |> Enum.filter(&(MLIR.Operation.name(&1) == "gpu.module"))
    |> Enum.each(&MLIR.Operation.get_and_update(&1, "targets", fn _ -> {nil, targets_attr} end))

    lowered
    |> Beaver.Composer.append("gpu-module-to-binary{format=#{format}}")
    |> Beaver.Composer.run!()
  end

  defp conversion_pass!([target | _]) do
    case target
         |> MLIR.CAPI.mlirAttributeGetDialect()
         |> MLIR.CAPI.mlirDialectGetNamespace()
         |> MLIR.to_string() do
      "nvvm" -> "convert-gpu-to-nvvm"
      "rocdl" -> "convert-gpu-to-rocdl"
      other -> raise ArgumentError, "unsupported GPU target attribute dialect: #{other}"
    end
  end

  defp conversion_pass!([]), do: raise(ArgumentError, "at least one target attribute is required")
end
