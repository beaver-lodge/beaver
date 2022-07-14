defmodule Beaver.MLIR.Managed.Location do
  alias Beaver.MLIR.CAPI

  @moduledoc """
  Getting and setting managed MLIR location
  """
  def get() do
    case Process.get(__MODULE__) do
      nil ->
        ctx = Beaver.MLIR.Managed.Context.get()
        location = CAPI.mlirLocationUnknownGet(ctx)
        set(location)

      managed ->
        managed
    end
  end

  def set(location) do
    Process.put(__MODULE__, location)
    location
  end

  def from_opts(opts) when is_list(opts) do
    location = opts[:loc]

    if location do
      location
    else
      get()
    end
  end
end
