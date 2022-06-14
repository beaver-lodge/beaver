defmodule Beaver.MLIR.Dialect.Builtin do
  defmacro module(call, do: block) do
    quote do
    end
  end

  @doc """
  Macro to create a module and insert ops into its body. region/1 shouldn't be called because region of one block will be created.
  """
  defmacro module(do: block) do
    quote do
      location = Beaver.MLIR.Managed.Location.get()
      module = Beaver.MLIR.CAPI.mlirModuleCreateEmpty(location)
      module_body_block = Beaver.MLIR.CAPI.mlirModuleGetBody(module)

      Beaver.MLIR.Block.under(module_body_block, fn ->
        unquote(block)
      end)

      module
    end
  end
end
