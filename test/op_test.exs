defmodule OpTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR

  test "get_and_update", %{ctx: ctx} do
    const =
      ~m"""
      %0 = arith.constant dense<42> : vector<4xi32>
      """
      |> Beaver.Deferred.resolve(ctx)
      |> MLIR.verify!()
      |> MLIR.Module.body()
      |> Beaver.Walker.operations()
      |> Enum.at(0)

    attr_str1 = "dense<1> : vector<4xi32>"
    attr_str2 = "dense<2> : vector<4xi32>"
    old_attr = const[:value]
    {attr, op} = get_and_update_in(const[:value], &{&1, ~a{#{attr_str1}}})
    assert MLIR.equal?(attr, old_attr)
    assert MLIR.equal?(op, const)
    assert const |> MLIR.to_string() =~ attr_str1

    # check deferred attribute
    old_attr = const[:value]

    {attr, op} =
      get_and_update_in(const[:value], &{&1, ~a{#{attr_str2}} |> Beaver.Deferred.resolve(ctx)})

    assert MLIR.equal?(attr, old_attr)
    assert MLIR.equal?(op, const)
    assert const |> MLIR.to_string() =~ attr_str2

    # check popping
    old_attr = const[:value]
    {popped, op} = pop_in(const[:value])
    assert MLIR.equal?(popped, old_attr)
    assert op[:value] == nil
  end

  test "incorrect argument", %{ctx: ctx} do
    alias MLIR.Dialect.MemRef

    assert_raise ArgumentError, ~r{Invalid argument.+:not_supported}s, fn ->
      mlir ctx: ctx do
        module do
          MemRef.global(:not_supported) >>> []
        end
      end
    end
  end

  test "builds an operation whose name is known at runtime", %{ctx: ctx} do
    MLIR.Context.allow_unregistered_dialects(ctx)
    operation = MLIR.Operation.builder("testing.dynamic")

    module =
      mlir ctx: ctx do
        module do
          value = operation.(label: ~a/producer/s) >>> ~t{i32}
          operation.(value, label: ~a/consumer/s) >>> []
        end
      end

    assert Enum.map(Beaver.Walker.operations(MLIR.Module.body(module)), &MLIR.Operation.name/1) ==
             ~w(testing.dynamic testing.dynamic)

    assert MLIR.verify!(module) == module
  end
end
