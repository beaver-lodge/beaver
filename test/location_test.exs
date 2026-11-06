defmodule Beaver.MLIR.LocationTest do
  use Beaver.Case, async: true

  alias Beaver.Changeset
  alias Beaver.MLIR

  test "reads and updates operation, result, and block argument provenance", %{ctx: ctx} do
    original = MLIR.Location.file(name: "original.ex", line: 1, ctx: ctx)
    replacement = MLIR.Location.file(name: "replacement.ex", line: 2, ctx: ctx)

    operation =
      %Changeset{name: "arith.constant", context: ctx, location: original}
      |> Changeset.add_result(MLIR.Type.i32(ctx: ctx))
      |> MLIR.Operation.create()

    block = MLIR.Block.create([MLIR.Type.i32(ctx: ctx)], [original])

    try do
      result = MLIR.Operation.result(operation, 0)
      argument = MLIR.Block.get_arg!(block, 0)

      assert MLIR.equal?(MLIR.Value.location(result), original)
      assert MLIR.equal?(MLIR.Value.location(argument), original)

      assert ^operation = MLIR.Operation.set_location(operation, replacement)
      assert MLIR.equal?(MLIR.Operation.location(operation), replacement)
      assert MLIR.equal?(MLIR.Value.location(result), replacement)

      assert ^argument = MLIR.Value.set_location(argument, replacement)
      assert MLIR.equal?(MLIR.Value.location(argument), replacement)

      assert_raise ArgumentError, "only block argument locations can be set", fn ->
        MLIR.Value.set_location(result, original)
      end
    after
      MLIR.Block.destroy(block)
      MLIR.Operation.destroy(operation)
    end
  end

  test "rejects provenance from a different context", %{ctx: ctx} do
    other_ctx = MLIR.Context.create()

    try do
      source = MLIR.Location.file(name: "other.ex", line: 1, ctx: other_ctx)

      assert_raise ArgumentError, "location belongs to a different MLIR context", fn ->
        MLIR.Location.resolve(source, ctx)
      end
    after
      MLIR.Context.destroy(other_ctx)
    end
  end
end
