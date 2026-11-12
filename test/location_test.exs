defmodule Beaver.MLIR.LocationTest do
  use Beaver.Case, async: true

  alias Beaver.Changeset
  alias Beaver.MLIR
  alias Beaver.MLIR.Location

  test "constructs and safely inspects source ranges", %{ctx: ctx} do
    location =
      Location.file_range(name: "source.ex", start_line: 4, start_column: 2, end_line: 6, end_column: 9, ctx: ctx)

    assert Location.kind(location) == :file_range
    assert Location.file_range?(location)

    assert Location.source_range(location) ==
             {:ok, %{file: "source.ex", start_line: 4, start_column: 2, end_line: 6, end_column: 9}}

    assert Location.name(location) == :error
  end

  test "retains nested named and call-site provenance", %{ctx: ctx} do
    callee = Location.file(name: "callee.ex", line: 11, column: 3, ctx: ctx)
    caller = Location.file(name: "caller.ex", line: 29, column: 7, ctx: ctx)
    named = Location.named("lowered", callee)
    call_site = Location.call_site(named, caller)
    assert Location.kind(named) == :name
    assert Location.name(named) == {:ok, "lowered"}
    assert {:ok, child} = Location.child(named)
    assert MLIR.equal?(child, callee)
    assert Location.kind(call_site) == :call_site
    assert {:ok, actual_callee} = Location.callee(call_site)
    assert {:ok, actual_caller} = Location.caller(call_site)
    assert MLIR.equal?(actual_callee, named)
    assert MLIR.equal?(actual_caller, caller)
    assert Location.metadata(call_site) == :error
  end

  test "retains fused children and metadata and round-trips its attribute", %{ctx: ctx} do
    source = Location.file(name: "origin.ex", line: 5, ctx: ctx)
    second_source = Location.file(name: "generated.ex", line: 8, ctx: ctx)
    metadata = MLIR.Attribute.string("rewrite-stage", ctx: ctx)
    single_source_fused = Location.with_metadata(source, metadata)
    fused = Location.with_metadata([source, second_source], metadata)
    assert Location.fused?(single_source_fused)
    assert Location.location_count(single_source_fused) == {:ok, 1}
    assert Location.kind(fused) == :fused
    assert Location.location_count(fused) == {:ok, 2}
    assert {:ok, [actual_source, actual_second_source]} = Location.locations(fused)
    assert MLIR.equal?(actual_source, source)
    assert MLIR.equal?(actual_second_source, second_source)
    assert {:ok, actual_metadata} = Location.metadata(fused)
    assert MLIR.equal?(actual_metadata, metadata)
    attribute = Location.attribute(fused)
    assert MLIR.Attribute.location?(attribute)
    assert attribute |> Location.from_attribute() |> MLIR.equal?(fused)
  end

  test "reads and updates operation, result, and block argument provenance", %{ctx: ctx} do
    original = Location.file(name: "original.ex", line: 1, ctx: ctx)
    replacement = Location.file(name: "replacement.ex", line: 2, ctx: ctx)

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
    MLIR.Context.with_context(fn other_ctx ->
      source = Location.file(name: "other.ex", line: 1, ctx: other_ctx)

      assert_raise ArgumentError, "location belongs to a different MLIR context", fn ->
        Location.named("wrong-context", source, ctx: ctx)
      end
    end)
  end
end
