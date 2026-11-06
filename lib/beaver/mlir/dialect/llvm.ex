defmodule Beaver.MLIR.Dialect.LLVM do
  @moduledoc "LLVM dialect operations. Construct types and attributes with MLIR.Type and MLIR.Attribute."
  use Beaver.MLIR.Dialect, dialect: "llvm", ops: Beaver.MLIR.Dialect.Registry.ops("llvm")

  defmacro func(call, opts) do
    quote do
      require Beaver.MLIR.Dialect.Func

      Beaver.MLIR.Dialect.Func.func_like(unquote(call), Beaver.MLIR.Dialect.LLVM.func(), unquote(opts))
    end
  end
end
