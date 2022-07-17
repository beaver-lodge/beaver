defmodule Beaver.Application do
  use Application
  @moduledoc false
  def start(_type, _args) do
    Supervisor.start_link(
      [
        Beaver.MLIR.Global.Context,
        Beaver.MLIR.Dialect.Registry,
        Beaver.MLIR.DSL.Op.Registry
      ],
      strategy: :one_for_one
    )
  end

  def start_phase(:load_dialect_modules, :normal, []) do
    for dialect_module <- Beaver.MLIR.Dialect.dialects() do
      for op_module <- apply(dialect_module, :ops, []) do
        apply(op_module, :register_op_prototype, [])
      end
    end

    :ok
  end
end
