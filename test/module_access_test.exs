defmodule ModuleAccessTest do
  use Beaver.Case, async: true

  @source "module {\n  func.func @test_func() -> i32 {\n    %0 = arith.constant 42 : i32\n    return %0 : i32\n  }\n}\n"

  test "module supports Access behavior", %{ctx: ctx} do
    opts = [ctx: ctx]
    {:ok, module} = MLIR.Module.create(@source, opts)
    # Test fetch with non-existent attribute
    assert :error = Access.fetch(module, "nonexistent_attribute")
    # Test get_and_update to add an attribute
    {current_value, updated_module} =
      Access.get_and_update(module, "test_attr", fn nil -> {nil, MLIR.Attribute.integer(MLIR.Type.i32(opts), 123)} end)

    assert current_value == nil
    assert {:ok, attr} = Access.fetch(updated_module, "test_attr")
    assert MLIR.Attribute.integer?(attr)
    # Test pop to remove the attribute
    {popped_attr, final_module} = Access.pop(updated_module, "test_attr")
    assert MLIR.Attribute.integer?(popped_attr)
    assert :error = Access.fetch(final_module, "test_attr")
  end

  test "module access works with bracket syntax", %{ctx: ctx} do
    opts = [ctx: ctx]

    {:ok, module} = MLIR.Module.create(@source, opts)
    # Test bracket syntax for getting attributes
    assert module["nonexistent"] == nil
    # Test bracket syntax for setting attributes
    module_with_attr = put_in(module["test_attr"], MLIR.Attribute.integer(MLIR.Type.i32(opts), 456))

    assert module_with_attr["test_attr"] != nil
  end
end
