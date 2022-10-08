defmodule DialectRegistryTest do
  use ExUnit.Case

  alias Beaver.MLIR.Dialect

  test "example from upstream with br" do
    assert not Enum.empty?(Dialect.Registry.dialects())
    assert not Enum.empty?(Dialect.Registry.ops("arith"))
    assert Dialect.Registry.ops("cf") == ["switch", "cond_br", "br", "assert"]

    tuples =
      for d <- Dialect.Registry.dialects(full: true) do
        {d, Dialect.Registry.normalize_dialect_name(d)}
      end
      |> Enum.sort()

    assert tuples ==
             Enum.sort([
               {"acc", "ACC"},
               {"affine", "Affine"},
               {"amdgpu", "AMDGPU"},
               {"amx", "AMX"},
               {"arith", "Arith"},
               {"arm_neon", "ArmNeon"},
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
               {"linalg", "Linalg"},
               {"llvm", "LLVM"},
               {"math", "Math"},
               {"memref", "MemRef"},
               {"ml_program", "MLProgram"},
               {"nvgpu", "NVGPU"},
               {"nvvm", "NVVM"},
               {"omp", "OMP"},
               {"pdl", "PDL"},
               {"pdl_interp", "PDLInterp"},
               {"quant", "Quant"},
               {"rocdl", "ROCDL"},
               {"scf", "SCF"},
               {"shape", "Shape"},
               {"sparse_tensor", "SparseTensor"},
               {"spirv", "Spirv"},
               {"tensor", "Tensor"},
               {"tosa", "TOSA"},
               {"transform", "Transform"},
               {"vector", "Vector"},
               {"x86vector", "X86Vector"}
             ])
  end

  test "ops are unique" do
    for d <- Dialect.Registry.dialects() do
      ops = Dialect.Registry.ops(d)
      assert Enum.uniq(ops) == ops
    end
  end
end
