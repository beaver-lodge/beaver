alias Beaver.MLIR
ctx = MLIR.Context.create()
{m, e} = AddENIF.init(ctx)
invoker = AddENIF.invoker(e)

Benchee.run(
  %{
    "jit.add" => fn {a, b} -> invoker.(a, b) end,
    "add" => fn {a, b} -> a + b end
  },
  inputs: %{
    "random num" => 10000
  },
  before_scenario: fn n ->
    a = :rand.uniform(n)
    b = :rand.uniform(n)
    {a, b}
  end
)

AddENIF.destroy(m, e)
MLIR.Context.destroy(ctx)
