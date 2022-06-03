defmodule Beaver.MLIR.CAPI.Registration do
  @deprecated "Use Beaver.MLIR.CAPI"
  @moduledoc false
  use Exotic.Library
  alias Beaver.MLIR.CAPI.IR.Context
  def mlirRegisterAllDialects(Context), do: :void

  defmodule DialectHandle do
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  def mlirDialectHandleRegisterDialect(DialectHandle, Context), do: :void
  def mlirGetDialectHandle__elixir__(), do: DialectHandle

  @doc """
  For some reason, this must be called before calling mlirRegisterAllDialects.
  Otherwise Elixir dialect won't be registered.
  """
  def register_elixir_dialect(ctx) do
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__elixir__(), ctx)
  end

  @native [
    mlirRegisterAllDialects: 1,
    mlirDialectHandleRegisterDialect: 2,
    mlirGetDialectHandle__elixir__: 0
  ]
  def load!(), do: Exotic.load!(__MODULE__, Beaver.MLIR.CAPI)
end
