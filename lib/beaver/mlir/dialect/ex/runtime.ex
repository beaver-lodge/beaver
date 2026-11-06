defmodule Beaver.MLIR.Dialect.Ex.Runtime do
  @moduledoc "Batata C ABI instance and rooted-value operations; statuses remain explicit."
  defmacro __using__(_) do
    quote do
      @ex_layer :runtime
      Beaver.MLIR.Dialect.Ex.Schema.defcops(:runtime)
      defop value_load(pointer = Beaver.MLIR.Type.llvm_pointer()), results: [value: ^runtime_value]
      defop value_store(value = ^runtime_value, pointer = Beaver.MLIR.Type.llvm_pointer()), results: []
    end
  end
end
