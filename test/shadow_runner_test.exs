defmodule Beaver.Shadow.RunnerTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR
  alias Beaver.Shadow.{Receipt, Runner}
  alias Beaver.MLIR.Transform.Schedule.DSL

  defmodule ClosedLoopSchedule do
    use DSL

    defschedule closed_loop do
      sequence "__transform_main", [_root >>> any_op()] do
        alternatives "route" do
          branch do
            _fast = knob("fast_tile", [8, 16])
          end

          branch do
            _slow = knob("slow_tile", [32, 64])
          end
        end
      end
    end
  end

  @payload """
  module {
    func.func @f(%arg0: i32) -> i32 {
      %0 = arith.constant 1 : i32
      %1 = arith.addi %arg0, %0 : i32
      return %1 : i32
    }
  }
  """

  setup do
    context = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(context) end)
    %{ctx: context}
  end

  test "records one receipt per candidate in deterministic order", %{ctx: ctx} do
    schedule = own(ClosedLoopSchedule.closed_loop(ctx: ctx))

    evaluator = fn resolved, candidate ->
      {:ok, resolved.digest,
       %{
         artifact: %{cache: :miss, lookup_key: resolved.digest, artifact_key: resolved.digest},
         trace: %{
           action_count: 1,
           tags: ["tile"],
           candidate_index: candidate.index,
           schedule_digest: resolved.digest
         }
       }}
    end

    run = Runner.run(@payload, schedule, evaluator: evaluator)

    assert run.input_digest ==
             :crypto.hash(:sha256, @payload) |> Base.encode16(case: :lower)

    assert length(run.receipts) == 4
    assert Enum.map(run.receipts, & &1.candidate.index) == [0, 1, 2, 3]

    assert Enum.map(run.receipts, & &1.candidate.choices) == [
             %{"route" => 0, "fast_tile" => 8},
             %{"route" => 0, "fast_tile" => 16},
             %{"route" => 1, "slow_tile" => 32},
             %{"route" => 1, "slow_tile" => 64}
           ]

    assert Enum.all?(run.receipts, &(&1.status == :ok))
    assert run.winner == hd(run.receipts)
  end

  test "failure candidates keep their failure category and no winner is picked" do
    context = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(context) end)
    schedule = own(ClosedLoopSchedule.closed_loop(ctx: context))

    run =
      Runner.run(@payload, schedule,
        evaluator: fn _resolved, _candidate -> {:error, :benchmark_failed} end
      )

    assert Enum.all?(run.receipts, &(&1.status == :failed))
    assert Enum.all?(run.receipts, &(&1.failure.kind == :evaluation_failure))
    assert run.winner == nil
  end

  test "receipts are JSON-serializable and replayable" do
    context = MLIR.Context.create()
    on_exit(fn -> MLIR.Context.destroy(context) end)
    schedule = own(ClosedLoopSchedule.closed_loop(ctx: context))

    run = Runner.run(@payload, schedule)
    winner = run.winner

    decoded =
      winner
      |> Receipt.encode!()
      |> Receipt.decode!()

    assert Receipt.identity(decoded) == Receipt.identity(winner)

    assert decoded.schedule.bytecode == winner.schedule.bytecode
    assert decoded.schedule.sequence == "__transform_main"
  end

  defp own(module) do
    on_exit(fn -> MLIR.Module.destroy(module) end)
    module
  end
end
