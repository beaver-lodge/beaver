defmodule Beaver.MLIR.Type do
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  def get(string, opts \\ []) when is_binary(string) do
    ctx = MLIR.Managed.Context.from_opts(opts)
    CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create(string))
  end
end
