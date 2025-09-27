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

  test "run mlir module defined by sigil", %{ctx: ctx} do
    import Beaver.Sigils
    import MLIR.{Transform, Conversion}

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
      |> MLIR.verify!()
      |> canonicalize
      |> cse
      |> Beaver.Composer.nested("func.func", convert_linalg_to_loops())
      |> Beaver.Composer.append("one-shot-bufferize")
      |> Beaver.Composer.nested(
        "func.func",
        ~w{convert-linalg-to-loops}
      )
      |> convert_scf_to_cf
      |> Beaver.Composer.append("finalize-memref-to-llvm")
      |> convert_cf_to_llvm()
      |> convert_arith_to_llvm()
      |> convert_func_to_llvm
      |> reconcile_unrealized_casts
      |> Beaver.Composer.run!()
      |> MLIR.ExecutionEngine.create!()

    arr = [0.112122112, 0.2123213, 10_020.9, 213_120.0, 0.2, 100.4]

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
             10_020.900390625,
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

  test "dynamic stride or offset" do
    assert Beaver.MLIR.Type.dynamic_stride_or_offset?(Beaver.MLIR.Type.dynamic_stride_or_offset())
    refute Beaver.MLIR.Type.static_stride_or_offset?(Beaver.MLIR.Type.dynamic_stride_or_offset())

    refute Beaver.MLIR.Type.dynamic_stride_or_offset?(1)
    assert Beaver.MLIR.Type.static_stride_or_offset?(1)

    assert Beaver.MLIR.Type.dynamic_stride_or_offset?(:dynamic)
    refute Beaver.MLIR.Type.static_stride_or_offset?(:dynamic)
  end
end
