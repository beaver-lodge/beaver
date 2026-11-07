defmodule Beaver.Application do
  use Application
  alias Beaver.MLIR
  @moduledoc false
  def start(_type, _args) do
    [
      MLIR.ThreadPool.child_spec(name: MLIR.ThreadPool.default_name()),
      MLIR.Pass.global_registrar_child_specs(),
      MLIR.RewritePattern.global_registrar_child_specs(),
      MLIR.Trait.global_registrar_child_specs(),
      MLIR.ExternalInterface.global_registrar_child_specs(),
      MLIR.ActionTracing.global_registrar_child_specs()
    ]
    |> List.flatten()
    |> Supervisor.start_link(strategy: :one_for_one)
  end
end
