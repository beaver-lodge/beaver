defmodule Adder do
  use Beaver.MLIR.Dialect.Elixir.Deprecated

  @mlir [
    add: 2,
    main: 0
  ]

  # NOTE: becuase there is no public API for type spec, so we use MLIR type attributes.
  # Downside is that this might force users to get to know MLIR types.
  # If we declare all Elixir types in MLIR and provide parsers this could make more sense.
  # Upside is that this gets attribute pasers and checks for free and provide more flexibility.
  # This also nullifies the compilicity of type specs with '|'
  @mlir_spec "(i64, i64) -> (i64)"
  def add(lhs, rhs) do
    lhs + rhs
  end

  @mlir_spec "() -> ()"
  def main() do
    a = 100
    _b = add(a, a)
  end
end

defmodule AdderTest do
  use ExUnit.Case

  test "test add two" do
    # Adder.main()
  end
end
