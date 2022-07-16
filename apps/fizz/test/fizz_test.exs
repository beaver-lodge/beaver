defmodule FizzTest do
  use ExUnit.Case, async: true
  doctest Fizz

  test "greets the world" do
    x = Fizz.MLIR.CAPI.mlirContextCreate() |> IO.inspect()
    Fizz.MLIR.CAPI.mlirContextDestroy(x) |> IO.inspect()
  end
end
