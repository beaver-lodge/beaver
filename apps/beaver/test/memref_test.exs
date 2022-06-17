defmodule MemRefTest do
  use ExUnit.Case
  alias Beaver.MLIR

  test "run mlir module defined by sigil" do
    import Beaver.MLIR.Sigils
    import MLIR.{Transforms, Conversion}
    alias Beaver.MLIR.ExecutionEngine.MemRefDescriptor

    jit =
      ~m"""
      #map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      func.func @generic_without_inputs(%arg0 : memref<?x?x?xf32>) attributes {llvm.emit_c_interface} {
        linalg.generic  {indexing_maps = [#map0],
                          iterator_types = ["parallel", "parallel", "parallel"]}
                        outs(%arg0 : memref<?x?x?xf32>) {
          ^bb0(%arg3: f32):
            %cst = arith.constant 1.000000e+00 : f32
            linalg.yield %cst : f32
          }
        return
      }
      """
      |> MLIR.Operation.verify!()
      |> canonicalize
      |> cse
      |> tosa_to_scf
      |> tosa_to_arith
      |> tosa_to_tensor()
      |> convert_tensor_to_linalg()
      |> MLIR.Pass.Composer.nested("func.func", [
        tosa_to_linalg(),
        linalg_fuse_elementwise_ops(),
        linalg_bufferize(),
        convert_linalg_to_loops(),
        lower_affine(),
        convert_math_to_llvm(),
        convert_scf_to_cf(),
        "arith-expand",
        "memref-expand"
      ])
      |> MLIR.Pass.Composer.nested("func.func", fn pm ->
        MLIR.Pass.pipeline!(pm, "tensor-bufferize")
      end)
      |> MLIR.Pass.Composer.pipeline("func-bufferize")
      |> MLIR.Pass.Composer.run!()
      |> convert_vector_to_llvm
      |> convert_memref_to_llvm
      |> convert_func_to_llvm
      |> reconcile_unrealized_casts
      |> MLIR.Pass.Composer.run!()
      |> MLIR.Operation.dump!()
      |> MLIR.ExecutionEngine.create!()

    arg0 =
      MemRefDescriptor.create(
        [1.1, 2.2, 3.3, 1.1, 2.2, 3.3] |> Enum.map(&Exotic.Value.get(:f32, &1)),
        [1, 2, 3],
        [0, 0, 0]
      )

    return = MLIR.ExecutionEngine.invoke!(jit, "generic_without_inputs", [arg0])

    for i <- 1..100 do
      # IO.inspect(i)
      # return = MLIR.ExecutionEngine.invoke!(jit, "generic_without_inputs", [arg0])
    end
  end
end
