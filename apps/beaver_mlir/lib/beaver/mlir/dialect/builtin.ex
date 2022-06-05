defmodule Beaver.MLIR.Dialect.Builtin do
  defmacro module(call, do: block) do
    quote do
    end
  end

  defmacro module(do: block) do
    quote do
      location = Beaver.MLIR.Managed.Location.get()
      module = Beaver.MLIR.CAPI.mlirModuleCreateEmpty(location)
      module_body_block = Beaver.MLIR.CAPI.mlirModuleGetBody(module)

      __module__insert_point__ = fn op ->
        Beaver.MLIR.CAPI.mlirBlockAppendOwnedOperation(module_body_block, op)
      end

      Beaver.MLIR.Managed.InsertionPoint.push(__module__insert_point__)
      unquote(block)
      Beaver.MLIR.Managed.InsertionPoint.pop()

      Beaver.MLIR.Managed.Block.clear_ids()

      if not Beaver.MLIR.Managed.InsertionPoint.empty?(),
        do: raise("insertion point should be cleared")

      module
    end
  end
end
