defmodule DialectRegistryTest do
  use ExUnit.Case, async: true

  alias Beaver.MLIR.Dialect

  test "dialect name helpers" do
    for d <- Dialect.Registry.dialects() do
      assert d == apply(Dialect, String.to_atom(d), [])
    end
  end

  test "example from upstream with br" do
    assert not Enum.empty?(Dialect.Registry.dialects())
    assert not Enum.empty?(Dialect.Registry.ops("arith"))
    assert Dialect.Registry.ops("cf") == ["assert", "br", "cond_br", "switch"]

    registered =
      for d <- Dialect.Registry.dialects(full: true) do
        {d, Dialect.Registry.normalize_dialect_name(d)}
      end
      |> MapSet.new()

    required =
      [
        {"acc", "ACC"},
        {"affine", "Affine"},
        {"amdgpu", "AMDGPU"},
        {"arith", "Arith"},
        {"arm_neon", "ArmNeon"},
        {"arm_sme", "ArmSME"},
        {"arm_sve", "ArmSVE"},
        {"async", "Async"},
        {"bufferization", "Bufferization"},
        {"builtin", "Builtin"},
        {"cf", "CF"},
        {"complex", "Complex"},
        {"dlti", "DLTI"},
        {"emitc", "EmitC"},
        {"func", "Func"},
        {"gpu", "GPU"},
        {"index", "Index"},
        {"irdl", "IRDL"},
        {"linalg", "Linalg"},
        {"llvm", "LLVM"},
        {"math", "Math"},
        {"memref", "MemRef"},
        {"ml_program", "MLProgram"},
        {"mpi", "MPI"},
        {"nvgpu", "NVGPU"},
        {"nvvm", "NVVM"},
        {"omp", "OMP"},
        {"pdl", "PDL"},
        {"pdl_interp", "PDLInterp"},
        {"ptr", "Ptr"},
        {"quant", "Quant"},
        {"rocdl", "ROCDL"},
        {"scf", "SCF"},
        {"shape", "Shape"},
        {"sparse_tensor", "SparseTensor"},
        {"spirv", "SPIRV"},
        {"tensor", "Tensor"},
        {"tosa", "TOSA"},
        {"transform", "Transform"},
        {"ub", "UB"},
        {"vector", "Vector"},
        {"xegpu", "XeGPU"},
        {"smt", "SMT"},
        {"xevm", "XeVM"},
        {"shard", "Shard"},
        {"wasmssa", "WasmSSA"}
      ]
      |> MapSet.new()

    optional =
      [
        {"amx", "AMX"},
        {"x86", "X86"},
        {"x86vector", "X86Vector"}
      ]
      |> MapSet.new()

    assert MapSet.difference(required, registered) == MapSet.new([])

    assert registered |> MapSet.difference(required) |> MapSet.difference(optional) ==
             MapSet.new([])
  end

  test "ops are unique" do
    Task.async_stream(Dialect.Registry.dialects(), fn d ->
      ops = Dialect.Registry.ops(d)
      assert Enum.uniq(ops) == ops
    end)
    |> Enum.to_list()
  end

  test "handler" do
    assert Dialect.handler(Dialect.arith())
    assert Dialect.handler(Dialect.llvm())
    refute Dialect.handler("foo")
  end
end
