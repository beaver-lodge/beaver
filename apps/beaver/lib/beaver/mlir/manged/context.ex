defmodule Beaver.MLIR.Global.Context do
  alias Beaver.MLIR
  use Agent

  def start_link([]) do
    # TODO: read opts from app config
    Agent.start_link(
      fn ->
        require Beaver.MLIR.Context
        ctx = MLIR.Context.create(allow_unregistered: true)

        MLIR.Context.allow_multi_thread ctx do
          MLIR.CAPI.mlirRegisterAllPasses()
        end

        ctx
      end,
      name: __MODULE__
    )
  end

  def get do
    Agent.get(__MODULE__, & &1)
  end
end

defmodule Beaver.MLIR.Managed.Context do
  @moduledoc """
  Getting and setting managed MLIR context
  """
  def get() do
    case Process.get(__MODULE__) do
      nil ->
        global = Beaver.MLIR.Global.Context.get()
        set(global)
        global

      managed ->
        managed
    end
  end

  def set(ctx) do
    Process.put(__MODULE__, ctx)
    ctx
  end

  def from_opts(opts) when is_list(opts) do
    ctx = opts[:ctx]

    if ctx do
      ctx
    else
      get()
    end
  end
end
