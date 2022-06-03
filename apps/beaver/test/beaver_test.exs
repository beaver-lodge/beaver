defmodule BeaverTest do
  use ExUnit.Case
  doctest Beaver

  test "greets the world" do
    assert Beaver.hello() == :world
  end
end
