defmodule Beaver.MLIR.Dialect.Elixir do
  @moduledoc """
  This module defines Elixir dialect to represent Elixir AST in MLIR. This is at the moment a placeholder for future development.
  """
  use Beaver.MLIR.Dialect,
    dialect: "elixir",
    ops: ~w{add}
end

defmodule Beaver.Exterior.Elixir do
  @moduledoc """
  This module defines the Exterior for Elixir dialect. This is at the moment a placeholder for future development.
  """
  alias Beaver.MLIR
  @behaviour Beaver.Exterior
  def register_dialect(ctx) do
    MLIR.CAPI.mlirGetDialectHandle__elixir__()
    |> MLIR.CAPI.mlirDialectHandleRegisterDialect(ctx)
  end
end
