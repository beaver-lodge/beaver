defmodule MemRefTest do
  use Beaver.Case, async: true

  @moduletag :smoke
  test "creation" do
    assert %Beaver.Native.Memory{} =
             Beaver.Native.Memory.new([1.1, 2.2, 3.3], type: Beaver.Native.F32, sizes: [3, 1])

    assert %Beaver.Native.Memory{} =
             Beaver.Native.Memory.new([1.1, 2.2, 3.3], type: Beaver.Native.F32, sizes: [])
  end

  test "strides" do
    assert [6, 3, 1] = Beaver.Native.Memory.dense_strides([1, 2, 3])
  end

  test "run mlir module defined by sigil", context do
    import Beaver.MLIR.Sigils
    import MLIR.{Transforms, Conversion}
    ctx = context[:ctx]

    jit =
      ~m"""
      #map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      func.func @generic_without_inputs(%arg0 : memref<?x?x?xf32>) attributes {llvm.emit_c_interface} {
        linalg.generic  {indexing_maps = [#map0],
                          iterator_types = ["parallel", "parallel", "parallel"]}
                        outs(%arg0 : memref<?x?x?xf32>) {
          ^bb0(%arg3: f32):
            %cst = arith.constant 2.000000e+00 : f32
            linalg.yield %cst : f32
          }
        return
      }
      """.(ctx)
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

    shape = [1, 2, 3]

    arg0 =
      Beaver.Native.Memory.new(
        arr,
        sizes: shape,
        type: Beaver.Native.F32
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
      |> Beaver.Native.Memory.aligned()
      |> Beaver.Native.OpaquePtr.to_binary(Integer.floor_div(32 * 6, 8))

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

    MLIR.ExecutionEngine.invoke!(jit, "generic_without_inputs", [
      Beaver.Native.Memory.descriptor_ptr(arg0)
    ])

    for _i <- 1..1000 do
      MLIR.ExecutionEngine.invoke!(jit, "generic_without_inputs", [
        Beaver.Native.Memory.descriptor_ptr(arg0)
      ])
    end

    <<
      a0::little-float-32,
      a1::little-float-32,
      a2::little-float-32,
      a3::little-float-32,
      a4::little-float-32,
      a5::little-float-32
    >> =
      arg0
      |> Beaver.Native.Memory.aligned()
      |> Beaver.Native.OpaquePtr.to_binary(Integer.floor_div(32 * 6, 8))

    assert [2.0, 2.0, 2.0, 2.0, 2.0, 2.0] ==
             [
               a0,
               a1,
               a2,
               a3,
               a4,
               a5
             ]
  end
end