defmodule Beaver.Application do
  use Application
  require Logger
  @moduledoc false
  def start(_type, _args) do
    Supervisor.start_link(
      [
        Beaver.MLIR.DSL.Op.Registry
      ],
      strategy: :one_for_one
    )
  end

  def start_phase(:load_dialect_modules, :normal, []) do
    Beaver.MLIR.CAPI.mlirRegisterAllPasses()

    for dialect_module <- Beaver.MLIR.Dialect.dialects() do
      Task.async(fn ->
        for op_module <- apply(dialect_module, :__ops__, []) do
          apply(op_module, :register_op_prototype, [])
        end
      end)
    end
    |> Task.await_many(10_000)

    Logger.debug("[Beaver] dialect modules loaded")
    :ok
  end
end
