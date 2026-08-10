defmodule Beaver.DeferredTest do
  use Beaver.Case, async: true

  alias Beaver.Deferred
  alias Beaver.MLIR

  test "builders distinguish deferred values from ordinary callbacks", %{ctx: ctx} do
    assert %Deferred{} = deferred = MLIR.Type.i32()
    assert %MLIR.Type{} = Deferred.resolve(deferred, ctx)
    assert %MLIR.Type{} = MLIR.Type.i32(ctx: ctx)
    assert %Deferred{} = MLIR.Type.vector([4], MLIR.Type.i32())
    assert {:ok, %MLIR.Type{}} = MLIR.Type.vector([4], MLIR.Type.i32(), ctx: ctx)
    assert %MLIR.Type{} = MLIR.Type.vector!([4], MLIR.Type.i32(), ctx: ctx)

    assert_raise ArgumentError,
                 "bare context resolvers are not deferred values; wrap the function with defer/1",
                 fn -> Deferred.resolve(fn _context -> :not_an_mlir_builder end, ctx) end
  end

  test "explicit invalid contexts fail at the builder boundary" do
    assert_raise ArgumentError, "expected :ctx to be an MLIR context, got: nil", fn ->
      MLIR.Type.i32(ctx: nil)
    end

    assert_raise ArgumentError, "expected an MLIR context, got: nil", fn ->
      Deferred.resolve(MLIR.Type.i32(), nil)
    end
  end

  test "resolution rejects eager and nested entities from another context", %{ctx: ctx} do
    other = MLIR.Context.create()

    try do
      type = MLIR.Type.i32(ctx: ctx)

      assert_raise ArgumentError, "type belongs to a different MLIR context", fn ->
        Deferred.resolve(type, other)
      end

      assert_raise ArgumentError, "type belongs to a different MLIR context", fn ->
        MLIR.Type.function([type], [], ctx: other)
      end
    after
      MLIR.Context.destroy(other)
    end
  end

  test "deferred success and error results have one normalization rule", %{ctx: ctx} do
    assert :resolved =
             Deferred.defer(fn _context -> {:ok, :resolved} end) |> Deferred.resolve(ctx)

    assert_raise ArgumentError, "failed", fn ->
      Deferred.defer(fn _context -> {:error, "failed"} end) |> Deferred.resolve(ctx)
    end
  end
end
