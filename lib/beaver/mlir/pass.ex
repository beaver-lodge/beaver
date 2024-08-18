defmodule Beaver.MLIR.Pass do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """

  alias Beaver.MLIR

  use Kinda.ResourceKind,
    fields: [handler: nil],
    forward_module: Beaver.Native

  @callback run(MLIR.Operation.t()) :: :ok | :error

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
      update_in(Agent.child_spec(fn -> nil end).start, fn {m, f, a} ->
        {m, f, a ++ [[name: @registrar]]}
      end),
      Task.child_spec(fn ->
        Agent.update(@registrar, fn _ -> Beaver.MLIR.CAPI.mlirRegisterAllPasses() end, :infinity)
      end)
    ]
  end
end
