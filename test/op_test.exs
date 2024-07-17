defmodule OpTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "get_and_update", test_context do
    const =
      ~m"""
      %0 = arith.constant dense<42> : vector<4xi32>
      """.(test_context[:ctx])
      |> MLIR.Operation.verify!()
      |> MLIR.Module.body()
      |> Beaver.Walker.operations()
      |> Enum.to_list()
      |> List.first()

    attr_str1 = "dense<1> : vector<4xi32>"
    attr_str2 = "dense<2> : vector<4xi32>"
    old_attr = const[:value]
    {attr, op} = get_and_update_in(const[:value], &{&1, ~a{#{attr_str1}}})
    assert MLIR.equal?(attr, old_attr)
    assert MLIR.equal?(op, const)
    assert const |> MLIR.to_string() =~ attr_str1

    # check deferred attribute
    old_attr = const[:value]
    {attr, op} = get_and_update_in(const[:value], &{&1, ~a{#{attr_str2}}.(test_context[:ctx])})
    assert MLIR.equal?(attr, old_attr)
    assert MLIR.equal?(op, const)
    assert const |> MLIR.to_string() =~ attr_str2

    # check popping
    old_attr = const[:value]
    {popped, op} = pop_in(const[:value])
    assert MLIR.equal?(popped, old_attr)
    assert op[:value] == nil
  end
end
