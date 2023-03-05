defmodule Beaver.MLIR.Location do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  @doc """
  Get a location of file line. Column is zero by default because in Elixir it is usually omitted.

  ## Examples
      iex> ctx = MLIR.Context.create()
      iex> MLIR.Location.file(name: "filename", line: 1, column: 1, ctx: ctx) |> MLIR.to_string()
      ~s{filename:1:1}
      iex> ctx |> MLIR.Context.destroy
  """

  def file(opts) do
    name = opts |> Keyword.fetch!(:name)
    line = opts |> Keyword.fetch!(:line)
    column = opts |> Keyword.get(:column, 0)

    Beaver.Deferred.from_opts(
      opts,
      &CAPI.mlirLocationFileLineColGet(&1, MLIR.StringRef.create(name), line, column)
    )
  end

  def unknown(opts \\ []) do
    Beaver.Deferred.from_opts(opts, &CAPI.mlirLocationUnknownGet/1)
  end
end
