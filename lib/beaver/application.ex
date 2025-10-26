defmodule Beaver.Application do
  use Application
  require Logger
  alias Beaver.MLIR
  @moduledoc false
  def start(_type, _args) do
    [
      MLIR.Pass.global_registrar_child_specs(),
      MLIR.RewritePattern.global_registrar_child_specs(),
      MLIR.Rewrite.thread_pool_child_spec()
    ]
    |> List.flatten()
    |> Supervisor.start_link(strategy: :one_for_one)
  end

  def stop([]) do
    MLIR.Rewrite.stop_thread_pool()
  end
end
