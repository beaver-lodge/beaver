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
      #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      func.func @generic_without_inputs(%arg0 : memref<1x2x3xf32>) attributes {llvm.emit_c_interface} {
        linalg.generic  {indexing_maps = [#map0],
                          iterator_types = ["parallel", "parallel", "parallel"]}
                        outs(%arg0 : memref<1x2x3xf32>) {
          ^bb0(%arg3: f32):
            %cst = arith.constant 2.000000e+00 : f32
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
      |> MLIR.ExecutionEngine.create!()

    arr = [0.112122112, 0.2123213, 10020.9, 213_120.0, 0.2, 100.4]

    arg0 =
      MemRefDescriptor.create(
        arr |> Enum.map(&Exotic.Value.get(:f32, &1)),
        [1, 2, 3],
        [0, 0, 0]
      )

    <<
      a0::little-float-32,
      a1::little-float-32,
      a2::little-float-32,
      a3::little-float-32,
      a4::little-float-32,
      a5::little-float-32
    >> =
      arg0
      |> Exotic.Value.fetch(
        Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(3),
        :allocated
      )
      |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 6, 8))

    assert [
             a0,
             a1,
             a2,
             a3,
             a4,
             a5
           ] == [
             0.11212211102247238,
             0.21232129633426666,
             10020.900390625,
             213_120.0,
             0.20000000298023224,
             100.4000015258789
           ]

    MLIR.ExecutionEngine.invoke!(jit, "generic_without_inputs", [Exotic.Value.get_ptr(arg0)])

    for _i <- 1..1000 do
      MLIR.ExecutionEngine.invoke!(jit, "generic_without_inputs", [Exotic.Value.get_ptr(arg0)])
    end

    assert arg0
           |> Exotic.Value.fetch(
             Beaver.MLIR.ExecutionEngine.MemRefDescriptor.struct_fields(3),
             :allocated
           )
           |> Exotic.Value.Ptr.read_as_binary(Integer.floor_div(32 * 6, 8)) ==
             <<0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64, 0, 0, 0, 64>>
  end
end
