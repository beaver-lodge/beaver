defmodule Beaver.MLIR.CAPI.FizzTest do
  use ExUnit.Case
  alias Beaver.MLIR.CAPI

  test "bool" do
    assert CAPI.bool(true) |> CAPI.to_term()
    assert not (CAPI.bool(false) |> CAPI.to_term())
  end
end
