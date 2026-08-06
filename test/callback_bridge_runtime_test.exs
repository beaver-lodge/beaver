defmodule Beaver.MLIR.CallbackBridgeRuntimeTest do
  use Beaver.Case, async: true

  test "TypeConverter callback projects a converted type from a native worker", %{ctx: ctx} do
    source = MLIR.Type.i32(ctx: ctx)
    target = MLIR.Type.i64(ctx: ctx)
    owner = self()

    converter =
      MLIR.TypeConverter.Callback.create(
        fn type ->
          send(owner, {:converted, type})
          target
        end,
        timeout: 1_000
      )

    assert {:ok, converted} = MLIR.TypeConverter.Callback.convert(converter, source)
    assert MLIR.equal?(converted, target)
    assert_receive {:converted, callback_type}
    assert MLIR.equal?(callback_type, source)
    assert :ok = MLIR.TypeConverter.Callback.destroy(converter)
  end

  test "TypeConverter callback supports declined and failure enum results", %{ctx: ctx} do
    source = MLIR.Type.i32(ctx: ctx)

    declined = MLIR.TypeConverter.Callback.create(fn _ -> :declined end, timeout: 1_000)
    assert {:error, :conversion_failed} = MLIR.TypeConverter.Callback.convert(declined, source)
    assert :ok = MLIR.TypeConverter.Callback.destroy(declined)

    failed =
      MLIR.TypeConverter.Callback.create(fn _ -> {:error, :unsupported} end, timeout: 1_000)

    assert {:error, :unsupported} = MLIR.TypeConverter.Callback.convert(failed, source)
    assert :ok = MLIR.TypeConverter.Callback.destroy(failed)
  end

  test "ConditionallySpeculatable fallback model dispatches through the context worker pool", %{
    ctx: ctx
  } do
    owner = self()

    MLIR.ConditionallySpeculatable.attach(
      ctx,
      "builtin.module",
      fn operation ->
        send(owner, {:speculated, operation})
        :recursively_speculatable
      end,
      timeout: 1_000
    )

    module = MLIR.Module.create!("module {}", ctx: ctx)
    operation = MLIR.Operation.from_module(module)

    assert :recursively_speculatable =
             MLIR.ConditionallySpeculatable.query(operation, timeout: 1_000)

    assert_receive {:speculated, callback_operation}
    assert MLIR.equal?(callback_operation, operation)
    MLIR.Module.destroy(module)
  end
end
