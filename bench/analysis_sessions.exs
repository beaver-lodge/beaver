alias Beaver.MLIR
alias MLIR.{AnalysisSession, Operation, Transform}

samples =
  case System.argv() do
    [count] -> String.to_integer(count)
    [] -> 100
  end

ctx = MLIR.Context.create()

module =
  MLIR.Module.create!(
    """
    module {
      func.func @compute(%x: i32) -> i32 {
        %a = arith.addi %x, %x : i32
        %b = arith.muli %a, %x : i32
        return %b : i32
      }
    }
    """,
    ctx: ctx
  )

schedule = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.yield
  }
}
"""

{:ok, prepared} = Transform.prepare(schedule, ctx)
session = AnalysisSession.create(module)
func = module |> MLIR.Module.body() |> Beaver.Walker.operations() |> Enum.at(0)
block = func |> Beaver.Walker.regions() |> Enum.at(0) |> Beaver.Walker.blocks() |> Enum.at(0)
[add, mul, _] = Enum.to_list(Beaver.Walker.operations(block))
queries = List.duplicate({:dominates, add, mul}, 100)

measure = fn fun ->
  for _ <- 1..20, do: fun.()
  times = for _ <- 1..samples, do: elem(:timer.tc(fun), 0)
  sorted = Enum.sort(times)

  %{
    median_us: Enum.at(sorted, div(samples, 2)),
    p95_us: Enum.at(sorted, min(samples - 1, div(samples * 95, 100)))
  }
end

try do
  results = %{
    samples: samples,
    build_mode: MLIR.CAPI.beaver_raw_execution_build_mode(),
    analysis_scalar_100:
      measure.(fn -> for _ <- 1..100, do: AnalysisSession.dominates?(session, add, mul) end),
    analysis_batch_100:
      measure.(fn ->
        %{results: results} = AnalysisSession.query(session, queries)
        true = Enum.all?(results)
      end),
    analysis_cold:
      measure.(fn ->
        AnalysisSession.with(
          module,
          &AnalysisSession.dead_after?(&1, Operation.result(add, 0), mul)
        )
      end),
    analysis_warm:
      measure.(fn -> AnalysisSession.dead_after?(session, Operation.result(add, 0), mul) end),
    transform_cold: measure.(fn -> {:ok, _} = Transform.execute(module, schedule) end),
    transform_prepared: measure.(fn -> {:ok, _} = Transform.execute(module, prepared) end)
  }

  IO.puts(JSON.encode!(results))
after
  AnalysisSession.close(session)
  Transform.close(prepared)
  MLIR.Module.destroy(module)
  MLIR.Context.destroy(ctx)
end
