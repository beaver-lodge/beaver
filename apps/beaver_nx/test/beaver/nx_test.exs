defmodule Beaver.NxTest do
  use ExUnit.Case
  doctest Beaver.Nx

  test "greets the world" do
    assert Beaver.Nx.hello() == :world
  end
end
