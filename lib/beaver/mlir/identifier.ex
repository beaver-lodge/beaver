defmodule Beaver.MLIR.Identifier do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  alias Beaver.MLIR

  def get(str, opts) do
    Beaver.Deferred.from_opts(opts, fn ctx -> MLIR.CAPI.mlirIdentifierGet(ctx, MLIR.StringRef.create(str)) end)
  end

  @doc "Converts `identifier` to an Elixir string."
  @spec to_string(t()) :: String.t()
  def to_string(%__MODULE__{} = id), do: MLIR.to_string(id)

  @doc "Returns whether two identifiers are equal."
  @spec equal?(t(), t()) :: boolean()
  def equal?(%__MODULE__{} = lhs, %__MODULE__{} = rhs), do: MLIR.equal?(lhs, rhs)
end
