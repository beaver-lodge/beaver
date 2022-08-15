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
      location = Beaver.MLIR.Managed.Location.get()
      import Beaver.MLIR.Sigils
      module = Beaver.MLIR.CAPI.mlirModuleCreateEmpty(location)

      for {name, attr} <- unquote(attrs) do
        name = name |> Beaver.MLIR.StringRef.create()

        module
        |> Beaver.MLIR.CAPI.mlirModuleGetOperation()
        |> Beaver.MLIR.CAPI.mlirOperationSetAttributeByName(name, attr)
      end

      module_body_block = Beaver.MLIR.CAPI.mlirModuleGetBody(module)

      mlir block: module_body_block do
        Beaver.MLIR.Dialect.Func.func printMemrefI32(
                                        sym_visibility: Beaver.MLIR.Attribute.string("private"),
                                        function_type: ~a"(memref<*xi32>) -> ()"
                                      ) do
          region do
          end
        end
      end

      Kernel.var!(beaver_internal_env_block) = module_body_block
      %Beaver.MLIR.CAPI.MlirBlock{} = Kernel.var!(beaver_internal_env_block)
      unquote(block)

      module
    end
  end
end
