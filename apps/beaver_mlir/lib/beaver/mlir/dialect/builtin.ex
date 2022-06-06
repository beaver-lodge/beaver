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

      Beaver.MLIR.Managed.Block.push(:module_body_block, module_body_block)

      Beaver.MLIR.Managed.InsertionPoint.push(fn op ->
        Beaver.MLIR.CAPI.mlirBlockAppendOwnedOperation(module_body_block, op)
      end)

      unquote(block)
      Beaver.MLIR.Managed.InsertionPoint.pop()
      Beaver.MLIR.Managed.Block.pop()

      Beaver.MLIR.Managed.Block.clear_ids()

      if not Beaver.MLIR.Managed.InsertionPoint.empty?(),
        do: raise("insertion point should be cleared")

      module
    end
  end
end
