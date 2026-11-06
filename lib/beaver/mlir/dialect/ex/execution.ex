defmodule Beaver.MLIR.Dialect.Ex.Execution do
  @moduledoc "Host-driven Batata process, step, event, reply and cancellation operations."
  defmacro __using__(_) do
    quote do
      @ex_layer :execution
      Beaver.MLIR.Dialect.Ex.Schema.defcops(:execution)
    end
  end
end
