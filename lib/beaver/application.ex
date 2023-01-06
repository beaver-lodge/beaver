defmodule Beaver.Application do
  use Application
  require Logger
  @moduledoc false
  def start(_type, _args) do
    Supervisor.start_link(
      [],
      strategy: :one_for_one
    )
  end

  def start_phase(:mlir_register_all_passes, :normal, []) do
    Beaver.MLIR.CAPI.mlirRegisterAllPasses()
    Logger.debug("[Beaver] all passes registered")
    :ok
  end
end
