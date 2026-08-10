defmodule Beaver.MLIR.LocationTest do
  use Beaver.Case, async: true

  alias Beaver.Changeset
  alias Beaver.MLIR

  test "constructs and safely inspects source ranges", %{ctx: ctx} do
    location =
      MLIR.Location.file_range(
        name: "source.ex",
        start_line: 4,
        start_column: 2,
        end_line: 6,
        end_column: 9,
        ctx: ctx
      )

    assert MLIR.Location.kind(location) == :file_range
    assert MLIR.Location.file_range?(location)

    assert MLIR.Location.source_range(location) ==
             {:ok,
              %{
                file: "source.ex",
                start_line: 4,
                start_column: 2,
                end_line: 6,
                end_column: 9
              }}

    assert MLIR.Location.name(location) == :error
  end

  test "retains nested named and call-site provenance", %{ctx: ctx} do
    callee = MLIR.Location.file(name: "callee.ex", line: 11, column: 3, ctx: ctx)
    caller = MLIR.Location.file(name: "caller.ex", line: 29, column: 7, ctx: ctx)
    named = MLIR.Location.named("lowered", callee)
    call_site = MLIR.Location.call_site(named, caller)

    assert MLIR.Location.kind(named) == :name
    assert MLIR.Location.name(named) == {:ok, "lowered"}
    assert {:ok, child} = MLIR.Location.child(named)
    assert MLIR.equal?(child, callee)

    assert MLIR.Location.kind(call_site) == :call_site
    assert {:ok, actual_callee} = MLIR.Location.callee(call_site)
    assert {:ok, actual_caller} = MLIR.Location.caller(call_site)
    assert MLIR.equal?(actual_callee, named)
    assert MLIR.equal?(actual_caller, caller)
    assert MLIR.Location.metadata(call_site) == :error
  end

  test "retains fused children and metadata and round-trips its attribute", %{ctx: ctx} do
    source = MLIR.Location.file(name: "origin.ex", line: 5, ctx: ctx)
    second_source = MLIR.Location.file(name: "generated.ex", line: 8, ctx: ctx)
    metadata = MLIR.Attribute.string("rewrite-stage", ctx: ctx)
    single_source_fused = MLIR.Location.with_metadata(source, metadata)
    fused = MLIR.Location.with_metadata([source, second_source], metadata)

    assert MLIR.Location.fused?(single_source_fused)
    assert MLIR.Location.location_count(single_source_fused) == {:ok, 1}

    assert MLIR.Location.kind(fused) == :fused
    assert MLIR.Location.location_count(fused) == {:ok, 2}
    assert {:ok, [actual_source, actual_second_source]} = MLIR.Location.locations(fused)
    assert MLIR.equal?(actual_source, source)
    assert MLIR.equal?(actual_second_source, second_source)
    assert {:ok, actual_metadata} = MLIR.Location.metadata(fused)
    assert MLIR.equal?(actual_metadata, metadata)

    attribute = MLIR.Location.attribute(fused)
    assert MLIR.Attribute.location?(attribute)
    assert attribute |> MLIR.Location.from_attribute() |> MLIR.equal?(fused)
  end

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
        MLIR.Location.named("wrong-context", source, ctx: ctx)
      end
    after
      MLIR.Context.destroy(other_ctx)
    end
  end
end
