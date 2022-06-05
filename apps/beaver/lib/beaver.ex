defmodule Beaver do
  @moduledoc """
  Documentation for `Beaver`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> Beaver.hello()
      :world

  """
  def hello do
    :world
  end

  @doc """
  This is a macro where Beaver's MLIR DSL expressions get transformed to MLIR API calls
  """
  defmacro mlir do
  end
end
