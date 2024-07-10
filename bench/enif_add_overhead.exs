alias Beaver.MLIR
ctx = MLIR.Context.create()
s = AddENIF.init(ctx)
invoker = &Beaver.ENIF.invoke(s.engine, "add", [&1, &2])

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

ENIFSupport.destroy(s)
MLIR.Context.destroy(ctx)
