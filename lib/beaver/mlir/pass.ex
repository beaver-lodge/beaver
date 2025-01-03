defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR
  use Kinda.ResourceKind, forward_module: Beaver.Native
  @callback run(MLIR.Operation.t()) :: any()

  defmacro __using__(opts) do
    quote do
      @behaviour MLIR.Pass
      Module.register_attribute(__MODULE__, :root_op, persist: true, accumulate: false)
      @root_op Keyword.get(unquote(opts), :on, "builtin.module")
    end
  end

  @registrar __MODULE__.GlobalRegistrar
  @doc """
  Ensure all passes are registered with the global registry.
  """
  def ensure_all_registered!() do
    :ok = Agent.get(@registrar, & &1, :infinity)
  end

  @doc false
  def global_registrar_child_specs() do
    [
      Supervisor.child_spec(Agent,
        start: {Agent, :start_link, [fn -> nil end, [name: @registrar]]}
      ),
      Task.child_spec(fn ->
        Agent.update(@registrar, fn _ -> Beaver.MLIR.CAPI.mlirRegisterAllPasses() end, :infinity)
      end)
    ]
  end
end
