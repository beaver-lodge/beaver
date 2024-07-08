defmodule EnifTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "populate enif functions", test_context do
    {m, e} = AddENIF.init(test_context[:ctx])
    invoker = AddENIF.invoker(e)
    assert 3 == invoker.(1, 2)
    assert 1 == invoker.(-1, 2)
    AddENIF.destroy(m, e)
  end

  test "query enif functions" do
    assert :enif_binary_to_term in Beaver.ENIF.functions()
  end
end
