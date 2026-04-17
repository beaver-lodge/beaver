defmodule Beaver.MLIR.Location do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR
  alias Beaver.Deferred

  use Kinda.ResourceKind, forward_module: Beaver.Native

  @doc """
  Get a location of file line. Column is zero by default because in Elixir it is usually omitted.

  ## Examples
      iex> ctx = MLIR.Context.create()
      iex> MLIR.Location.file(name: "filename", line: 1, column: 1, ctx: ctx) |> MLIR.to_string()
      ~s{filename:1:1}
      iex> MLIR.Context.destroy(ctx)
  """

  @type file_opts() :: [
          name: String.t(),
          line: integer(),
          column: integer() | nil,
          ctx: Deferred.context_arg()
        ]
  @type env_opts() :: [column: integer() | nil, ctx: Deferred.context_arg()]
  @type env_like() :: %{optional(:file) => String.t() | nil, optional(:line) => integer() | nil}
  @spec file(file_opts()) :: Deferred.contextual(MLIR.Location.t())
  def file(opts) do
    name = opts |> Keyword.fetch!(:name)
    line = opts |> Keyword.fetch!(:line)
    column = opts |> Keyword.get(:column, 0)

    Beaver.Deferred.from_opts(
      opts,
      &CAPI.mlirLocationFileLineColGet(&1, MLIR.StringRef.create(name), line, column)
    )
  end

  @doc """
  Create an MLIR location from `Macro.Env`
  """
  @spec from_env(env_like(), env_opts()) :: Deferred.contextual(MLIR.Location.t())
  def from_env(env, opts \\ []) when is_map(env) do
    name = env |> Map.get(:file, "nofile") |> to_string()
    line = env |> Map.get(:line)
    line = if is_integer(line), do: line, else: 0

    file(env_file_opts(name, line, opts))
  end

  @spec unknown(Deferred.opts()) :: Deferred.contextual(MLIR.Location.t())
  def unknown(opts \\ []) do
    Beaver.Deferred.from_opts(opts, &CAPI.mlirLocationUnknownGet/1)
  end

  @spec env_file_opts(String.t(), integer(), env_opts()) :: file_opts()
  defp env_file_opts(name, line, opts) do
    opts
    |> Keyword.put(:name, name)
    |> Keyword.put(:line, line)
  end
end
