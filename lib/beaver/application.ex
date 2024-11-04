defmodule Beaver.Application do
  use Application
  require Logger
  @moduledoc false
  def start(_type, _args) do
    Supervisor.start_link(
      Beaver.MLIR.Pass.global_registrar_child_specs() ++
        [
          {DynamicSupervisor, name: Beaver.Composer.DynamicSupervisor, strategy: :one_for_one},
          {Registry, keys: :unique, name: Beaver.Composer.Registry}
        ],
      strategy: :one_for_one
    )
  end
end
