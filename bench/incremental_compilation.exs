alias Beaver.MLIR

source = """
module {
  func.func @add(%lhs: i32, %rhs: i32) -> i32 {
    %result = arith.addi %lhs, %rhs : i32
    return %result : i32
  }
}
"""

{:ok, cold_cache} = MLIR.CompilationCache.Memory.start_link()
{:ok, warm_cache} = MLIR.CompilationCache.Memory.start_link()
cold_cache = {:memory, cold_cache}
warm_cache = {:memory, warm_cache}
pipeline = "canonicalize,cse"

MLIR.CompilationRuntime.compile!(source, cache: warm_cache, pipeline: pipeline)

Benchee.run(
  %{
    "cold compile" => fn ->
      MLIR.CompilationRuntime.invalidate(cold_cache)
      MLIR.CompilationRuntime.compile!(source, cache: cold_cache, pipeline: pipeline)
    end,
    "warm compile" => fn ->
      MLIR.CompilationRuntime.compile!(source, cache: warm_cache, pipeline: pipeline)
    end
  },
  warmup: 2,
  time: 5,
  memory_time: 2
)
