defmodule Beaver.Exterior.Elixir do
  alias Beaver.MLIR
  @behaviour Beaver.Exterior
  def register_dialect(ctx) do
    MLIR.CAPI.mlirDialectHandleRegisterDialect(
      MLIR.CAPI.mlirGetDialectHandle__elixir__(),
      ctx
    )
  end
end
