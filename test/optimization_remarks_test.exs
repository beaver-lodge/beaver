defmodule OptimizationRemarksTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR.{Context, Location, OptimizationRemarks}
  @moduletag :optimization_remarks

  test "copies payloads, filters categories, and preserves unrelated messages", %{ctx: ctx} do
    ref = OptimizationRemarks.attach(ctx, filters: [passed: "beaver"])
    loc = Location.unknown(ctx: ctx)

    assert OptimizationRemarks.emit(loc, :passed, "fold",
             args: [value: "你好\0quoted\""],
             function: "work"
           )

    refute OptimizationRemarks.emit(loc, :missed, "filtered")
    refute OptimizationRemarks.emit(loc, :passed, "filtered", category: "other")
    send(self(), :unrelated)
    OptimizationRemarks.finalize(ctx)

    assert [
             %{
               "kind" => "passed",
               "name" => "fold",
               "function" => "work",
               "args" => args,
               "location" => location
             }
           ] = OptimizationRemarks.drain(ref)

    assert ["value", "你好\0quoted\""] in args
    assert is_binary(location)
    assert_receive :unrelated
    assert OptimizationRemarks.drain(ref) == []
    refute OptimizationRemarks.enabled?(ctx)
  end

  test "final policy flushes on explicit finalization and context destruction" do
    for destroy <- [false, true] do
      ctx = Context.create()
      ref = OptimizationRemarks.attach(ctx, policy: :final)
      assert OptimizationRemarks.emit(Location.unknown(ctx: ctx), :analysis, "final")
      assert OptimizationRemarks.drain(ref) == []
      if destroy, do: Context.destroy(ctx), else: OptimizationRemarks.finalize(ctx)
      assert [%{"name" => "final"}] = OptimizationRemarks.drain(ref)
      unless destroy, do: Context.destroy(ctx)
    end
  end

  test "failed attachment preserves the original engine and releases the rejected receiver", %{
    ctx: ctx
  } do
    assert_raise ArgumentError, ~r/cannot enable/, fn ->
      OptimizationRemarks.attach(ctx, filters: [all: "["])
    end

    refute OptimizationRemarks.enabled?(ctx)
    ref = OptimizationRemarks.attach(ctx)
    assert_raise ArgumentError, ~r/cannot enable/, fn -> OptimizationRemarks.attach(ctx) end
    assert OptimizationRemarks.emit(Location.unknown(ctx: ctx), :failure, "original")
    OptimizationRemarks.finalize(ctx)
    OptimizationRemarks.finalize(ctx)
    assert [%{"name" => "original"}] = OptimizationRemarks.drain(ref)
  end

  test "worker threads deliver owned messages without waiting for BEAM callbacks", %{ctx: ctx} do
    ref = OptimizationRemarks.attach(ctx)
    loc = Location.unknown(ctx: ctx)

    1..100
    |> Task.async_stream(fn n -> assert OptimizationRemarks.emit(loc, :passed, "worker-#{n}") end,
      max_concurrency: 16
    )
    |> Enum.each(fn {:ok, true} -> :ok end)

    OptimizationRemarks.finalize(ctx)

    assert OptimizationRemarks.drain(ref) |> Enum.map(& &1["name"]) |> Enum.uniq() |> length() ==
             100
  end

  test "a departed receiver does not prevent context cleanup", %{ctx: ctx} do
    task = Task.async(fn -> OptimizationRemarks.attach(ctx) end)
    assert is_reference(Task.await(task))
    assert OptimizationRemarks.emit(Location.unknown(ctx: ctx), :passed, "orphan")
    assert :ok = OptimizationRemarks.finalize(ctx)
    refute OptimizationRemarks.enabled?(ctx)
  end
end
