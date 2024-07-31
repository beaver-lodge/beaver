alias Beaver.MLIR
ctx = MLIR.Context.create()
s = AddENIF.init(ctx)
invoker = &Beaver.ENIF.invoke(s.engine, "add", [&1, &2])
invoker_cpu = &Beaver.ENIF.invoke(s.engine, "add", [&1, &2], dirty: :cpu_bound)
invoker_io = &Beaver.ENIF.invoke(s.engine, "add", [&1, &2], dirty: :io_bound)

a = b = 100
invoker.(a, b)
invoker_cpu.(a, b)
invoker_io.(a, b)

Benchee.run(
  %{
    "jit.add" => fn {a, b} -> invoker.(a, b) end,
    "jit.add [cpu_bound]" => fn {a, b} -> invoker_cpu.(a, b) end,
    "jit.add [io_bound]" => fn {a, b} -> invoker_io.(a, b) end,
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
