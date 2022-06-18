defmodule DialectRegistryTest do
  use ExUnit.Case

  alias Beaver.MLIR.Dialect

  test "example from upstream with br" do
    assert not Enum.empty?(Dialect.Registry.dialects())
    assert not Enum.empty?(Dialect.Registry.ops("arith"))
    assert Dialect.Registry.ops("cf") == ["switch", "cond_br", "br", "assert"]

    tuples =
      for d <- Dialect.Registry.dialects() do
        {d, Dialect.Registry.normalize_dialect_name(d)}
      end

    assert tuples ==
             [
               {"affine", "Affine"},
               {"builtin", "Builtin"},
               {"tosa", "TOSA"},
               {"nvgpu", "NVGPU"},
               {"func", "Func"},
               {"gpu", "GPU"},
               {"pdl_interp", "PDLInterp"},
               {"x86vector", "X86Vector"},
               {"omp", "OMP"},
               {"emitc", "EmitC"},
               {"sparse_tensor", "SparseTensor"},
               {"memref", "MemRef"},
               {"amdgpu", "AMDGPU"},
               {"async", "Async"},
               {"arith", "Arith"},
               {"llvm", "LLVM"},
               {"linalg", "Linalg"},
               {"vector", "Vector"},
               {"cf", "CF"},
               {"transform", "Transform"},
               {"ml_program", "MLProgram"},
               {"tensor", "Tensor"},
               {"complex", "Complex"},
               {"amx", "AMX"},
               {"bufferization", "Bufferization"},
               {"arm_neon", "ArmNeon"},
               {"spv", "SPV"},
               {"math", "Math"},
               {"quant", "Quant"},
               {"arm_sve", "ArmSVE"},
               {"rocdl", "ROCDL"},
               {"acc", "ACC"},
               {"shape", "Shape"},
               {"pdl", "PDL"},
               {"nvvm", "NVVM"},
               {"scf", "SCF"}
             ]
  end
end
