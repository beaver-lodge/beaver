defmodule OpTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils

  test "get_and_update", test_context do
    constant =
      ~m"""
      %0 = arith.constant dense<42> : vector<4xi32>
      """.(test_context[:ctx])
      |> MLIR.Operation.verify!()
      |> MLIR.Module.body()
      |> Beaver.Walker.operations()
      |> Enum.to_list()
      |> List.first()

    new_attr_str1 = "dense<1> : vector<4xi32>"
    new_attr_str2 = "dense<2> : vector<4xi32>"
    old_attr = constant[:value]
    {attr, op} = get_and_update_in(constant[:value], &{&1, ~a{#{new_attr_str1}}})
    assert MLIR.equal?(attr, old_attr)
    assert MLIR.equal?(op, constant)
    assert constant |> MLIR.to_string() =~ new_attr_str1

    old_attr = constant[:value]

    {attr, op} =
      get_and_update_in(constant[:value], &{&1, ~a{#{new_attr_str2}}.(test_context[:ctx])})

    assert MLIR.equal?(attr, old_attr)
    assert MLIR.equal?(op, constant)
    assert constant |> MLIR.to_string() =~ new_attr_str2
  end
end
