defmodule Beaver.Application do
  use Application
  require Logger
  @moduledoc false
  def start(_type, _args) do
    [Beaver.MLIR.Pass.global_registrar_child_specs()]
    |> List.flatten()
    |> Supervisor.start_link(strategy: :one_for_one)
  end
end
