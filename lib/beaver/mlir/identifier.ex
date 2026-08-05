defmodule Beaver.MLIR.Identifier do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  alias Beaver.MLIR

  def get(str, opts) do
    Beaver.Deferred.from_opts(
      opts,
      fn ctx ->
        MLIR.CAPI.mlirIdentifierGet(ctx, MLIR.StringRef.create(str))
      end
    )
  end
end
