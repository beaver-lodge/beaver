defmodule Beaver.MLIR.Application do
  @moduledoc false
  def start(_type, _args) do
    Supervisor.start_link(
      [
        Beaver.MLIR.CAPI.Managed,
        Beaver.MLIR.Global.Context,
        Beaver.MLIR.Dialect.Registry
      ],
      strategy: :one_for_one
    )
  end
end
