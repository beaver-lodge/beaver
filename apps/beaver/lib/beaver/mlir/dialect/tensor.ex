defmodule Beaver.MLIR.Dialect.Tensor do
  alias Beaver.MLIR.{Attribute, Type}

  use Beaver.MLIR.Dialect,
    dialect: "tensor",
    ops: Beaver.MLIR.Dialect.Registry.ops("tensor")

  def reassociation(list) do
    for pair <- list do
      case pair do
        [src, dst] = pair when is_integer(src) and is_integer(dst) ->
          pair
          |> Enum.map(&Attribute.integer(Type.i64(), &1))
          |> Attribute.array()

        _ ->
          raise "expect a pair [integer(), integer()]"
      end
    end
    |> Attribute.array()
  end
end
