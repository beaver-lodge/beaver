defmodule Beaver.TestSupport.IR do
  @moduledoc false

  alias Beaver.Changeset
  alias Beaver.MLIR

  def single_operation_module(%MLIR.Context{} = ctx, operation_name)
      when is_binary(operation_name) do
    location = MLIR.Location.unknown(ctx: ctx)
    module = MLIR.Module.empty(location)

    operation =
      %Changeset{name: operation_name, context: ctx, location: location}
      |> MLIR.Operation.create()

    MLIR.Block.append(MLIR.Module.body(module), operation)
    module
  end
end
