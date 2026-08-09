defmodule Beaver.Shadow.WasmTest do
  use Beaver
  use Beaver.Case, async: true

  alias Beaver.MLIR
  alias Beaver.Shadow.{Receipt, Wasm}
  alias Beaver.MLIR.Transform.Schedule.DSL

  defmodule ClosedLoopSchedule do
    use DSL

    defschedule closed_loop do
      sequence "__transform_main", [_root >>> any_op()] do
        alternatives "route" do
          branch do
            _fast = knob("opt", ["-O1", "-O2"])
          end

          branch do
            _slow = knob("nostdlib", [true, false])
          end
        end
      end
    end
  end

  @payload """
  module {
    func.func @add(%arg0: i32, %arg1: i32) -> i32 {
      %0 = arith.addi %arg0, %arg1 : i32
      return %0 : i32
    }
  }
  """

  setup do
    context = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(context) end)
    %{ctx: context}
  end

  test "records run evidence through node", %{ctx: ctx} do
    schedule = ClosedLoopSchedule.closed_loop(ctx: ctx)

    run =
      Wasm.run(@payload, schedule,
        source: @payload,
        entry: "add",
        args: [2, 3]
      )

    assert length(run.receipts) == 4
    assert Enum.all?(run.receipts, &(&1.status == :ok))

    winner = run.winner
    assert winner.user_metadata.value == "5"
    assert winner.trace.tags == ["wasm.run"]
    assert winner.user_metadata.runtime.runner == "node"
    assert is_float(winner.user_metadata.durations.run_ms)
    assert is_integer(winner.user_metadata.durations.total_native)
    assert Enum.any?(winner.user_metadata.exports, &(&1.name == "add"))

    assert Receipt.identity(winner) ==
             winner |> Receipt.encode!() |> Receipt.decode!() |> Receipt.identity()
  end

  test "degrades to a recorded failure without a wasm runtime", %{ctx: ctx} do
    schedule = ClosedLoopSchedule.closed_loop(ctx: ctx)

    run =
      Wasm.run(@payload, schedule,
        source: @payload,
        runner: "/nonexistent/node"
      )

    assert Enum.all?(run.receipts, &(&1.status == :failed))
    assert Enum.all?(run.receipts, &match?(%{kind: :evaluation_failure}, &1.failure))
    assert Enum.all?(run.receipts, &match?({:wasm_unavailable, _}, &1.failure.reason))
    assert run.winner == nil
  end
end
