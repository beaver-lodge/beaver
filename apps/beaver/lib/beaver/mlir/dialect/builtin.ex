defmodule Beaver.MLIR.Dialect.Builtin do
  alias Beaver.MLIR.Dialect

  use Beaver.MLIR.Dialect.Generator,
    dialect: "builtin",
    ops: Dialect.Registry.ops("builtin"),
    skips: ~w{module}

  defmodule Module do
    use Beaver.DSL.Op.Prototype, op_name: "builtin.module"
  end

  defmacro module(_call, do: _block) do
    raise "TODO: support module with symbol"
  end

  @doc """
  Macro to create a module and insert ops into its body. region/1 shouldn't be called because region of one block will be created.
  """
  defmacro module(do: block) do
    quote do
      location = Beaver.MLIR.Managed.Location.get()
      module = Beaver.MLIR.CAPI.mlirModuleCreateEmpty(location)
      module_body_block = Beaver.MLIR.CAPI.mlirModuleGetBody(module)

      Kernel.var!(beaver_internal_env_block) = module_body_block
      %Beaver.MLIR.CAPI.MlirBlock{} = Kernel.var!(beaver_internal_env_block)
      unquote(block)

      module
    end
  end
end
