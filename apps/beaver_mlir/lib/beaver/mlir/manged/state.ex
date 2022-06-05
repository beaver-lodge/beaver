defmodule Beaver.MLIR.Managed.Terminator do
  @moduledoc """
  Pending terminator creators
  """

  def defer(creator) when is_function(creator) do
    old = Process.delete(__MODULE__) || []
    new = old ++ [creator]
    Process.put(__MODULE__, new)
    new
  end

  def resolve() do
    all = Process.delete(__MODULE__) || []

    for f <- all do
      f.()
    end
  end
end
