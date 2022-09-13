defmodule Beaver.MLIR.Dialect.Builtin do
  alias Beaver.MLIR.Dialect

  use Beaver.MLIR.Dialect,
    dialect: "builtin",
    ops: Dialect.Registry.ops("builtin"),
    skips: ~w{module}

  defmodule Module do
    use Beaver.DSL.Op.Prototype, op_name: "builtin.module"
  end

  @doc """
  Macro to create a module and insert ops into its body. region/1 shouldn't be called because region of one block will be created.
  """
  defmacro module(attrs \\ [], do: block) do
    quote do
      ctx = Beaver.MLIR.__CONTEXT__()
      location = Beaver.MLIR.Location.unknown(ctx: ctx)
      import Beaver.MLIR.Sigils
      module = Beaver.MLIR.CAPI.mlirModuleCreateEmpty(location)

      for {name, attr} <- unquote(attrs) do
        name = name |> Beaver.MLIR.StringRef.create()

        attr = Beaver.Deferred.create(attr, ctx)

        module
        |> Beaver.MLIR.CAPI.mlirModuleGetOperation()
        |> Beaver.MLIR.CAPI.mlirOperationSetAttributeByName(name, attr)
      end

      module_body_block = Beaver.MLIR.CAPI.mlirModuleGetBody(module)

      Kernel.var!(beaver_internal_env_block) = module_body_block
      %Beaver.MLIR.CAPI.MlirBlock{} = Kernel.var!(beaver_internal_env_block)
      unquote(block)

      module
    end
  end
end
