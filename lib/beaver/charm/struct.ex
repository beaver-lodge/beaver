defmodule Beaver.Charm.Struct do
  use Beaver
  defstruct field_types: [], ctx: nil

  defmacro __using__(fields) do
    quote do
      def __charmtype__ do
        unquote(fields)
      end
    end
  end

  def update_field(%Beaver.Charm.Struct{ctx: ctx, field_types: field_types}, field, cb) do
    mlir ctx: ctx do
      module do
        cb.(ctx, Beaver.Env.block(), field_types[field])
      end
    end
    |> MLIR.Operation.verify!()
  end
end
