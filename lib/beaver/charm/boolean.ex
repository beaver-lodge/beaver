defmodule Charm.Boolean do
  use Beaver
  alias MLIR.Type
  alias Beaver.Charm
  alias MLIR.Dialect.Index
  alias Beaver.Charm.Struct
  use Charm.Struct, value: Type.i1()

  defstruct value: false

  def __init__(%Beaver.Charm.Struct{ctx: _ctx} = this) do
    Struct.update_field(this, :value, fn ctx, block, type ->
      mlir block: block, ctx: ctx do
        Index.bool_constant(value: ~a{false}) >>> type
      end
    end)
  end

  def __init__() do
    ctx = MLIR.Context.create()
    this = %Beaver.Charm.Struct{field_types: __charmtype__(), ctx: ctx}
    __init__(this)
  end
end
