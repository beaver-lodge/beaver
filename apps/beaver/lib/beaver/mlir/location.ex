defmodule Beaver.MLIR.Location do
  alias Beaver.MLIR.CAPI

  # only file and line because usually there is no column in elixir source info
  # TODO: this fails when using Beaver.MLIR.CAPI
  def get!(ctx, file, line) do
    CAPI.mlirLocationFileLineColGet(
      ctx,
      file,
      line,
      0
    )
  end

  def get!(ctx, file: file, line: line) do
    get!(ctx, file, line)
  end
end
