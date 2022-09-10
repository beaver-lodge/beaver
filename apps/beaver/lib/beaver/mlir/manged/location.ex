defmodule Beaver.MLIR.Managed.Location do
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR

  @moduledoc """
  Getting and setting managed MLIR location
  """
  def get(opts \\ []) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    location = CAPI.mlirLocationUnknownGet(ctx)
    set(location)
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
