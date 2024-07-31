alias Beaver.MLIR
ctx = MLIR.Context.create()
s = AddENIF.init(ctx)
invoker = &Beaver.ENIF.invoke(s.engine, "add", [&1, &2])
invoker_cpu = &Beaver.ENIF.invoke(s.engine, "add", [&1, &2], dirty: :cpu_bound)
invoker_io = &Beaver.ENIF.invoke(s.engine, "add", [&1, &2], dirty: :io_bound)

a = 123
b = 456
sum = a + b
^sum = invoker_cpu.(a, b)
^sum = invoker_io.(a, b)
^sum = invoker.(a, b)
ENIFSupport.destroy(s)
MLIR.Context.destroy(ctx)
