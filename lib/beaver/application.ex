defmodule Beaver.Application do
  use Application
  require Logger
  @moduledoc false
  def start(_type, _args) do
    Supervisor.start_link(Beaver.MLIR.Pass.global_registrar_child_specs(), strategy: :one_for_one)
  end
end
