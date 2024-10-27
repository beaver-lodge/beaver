defmodule TosaTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR.Dialect.{Func, TOSA}
  require Func
  alias Beaver.Native
  import MLIR.{Transforms, Conversion}

  def test_lower_to_llvm(op) do
    op
    |> MLIR.Pass.Composer.nested("func.func", [convert_vector_to_scf(), convert_linalg_to_loops()])
    |> lower_affine()
    |> convert_scf_to_cf()
    |> canonicalize()
    |> cse()
    |> MLIR.Pass.Composer.append("convert-vector-to-llvm{reassociate-fp-reductions}")
    |> MLIR.Pass.Composer.nested("func.func", convert_math_to_llvm())
    |> MLIR.Pass.Composer.append("expand-strided-metadata")
    |> lower_affine()
    |> MLIR.Pass.Composer.append("finalize-memref-to-llvm")
    |> convert_func_to_llvm
    |> convert_index_to_llvm
    |> reconcile_unrealized_casts
  end

  test "generate and run tosa", %{ctx: ctx} do
    ir =
      mlir ctx: ctx do
        module do
          Func.func test_multi_broadcast(
                      function_type: ~a"(tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>"
                    ) do
            region do
              block _entry(
                      arg0 >>> Type.ranked_tensor([:dynamic, :dynamic], Type.f32()),
                      arg1 >>> Type.ranked_tensor([:dynamic, :dynamic], Type.f32())
                    ) do
                v0 = TOSA.add(arg0, arg1) >>> Type.ranked_tensor([:dynamic, :dynamic], Type.f32())

                v0 =
                  TOSA.mul(v0, arg1, {:shift, ~a{0 : i8}}) >>>
                    Type.ranked_tensor([:dynamic, :dynamic], Type.f32())

                Func.return(v0) >>> []
              end
            end
          end
        end
      end

    ir
    |> MLIR.Operation.from_module()
    |> MLIR.Pass.Composer.nested("func.func", [
      tosa_to_linalg_named(),
      tosa_to_linalg(),
      tosa_to_arith()
    ])
    |> MLIR.Pass.Composer.append("one-shot-bufferize{bufferize-function-boundaries}")
    |> MLIR.Pass.Composer.append("buffer-deallocation-pipeline")
    |> MLIR.Pass.Composer.nested("func.func", "llvm-request-c-wrappers")
    |> test_lower_to_llvm
    |> MLIR.Pass.Composer.run!()

    jit = ir |> MLIR.ExecutionEngine.create!()

    arg0 =
      Native.Memory.new(
        [1.1, 2.2, 3.3],
        type: Native.F32,
        sizes: [1, 3],
        strides: [3, 1]
      )

    arg1 =
      Native.Memory.new(
        [1.1, 2.2],
        type: Native.F32,
        sizes: [2, 1],
        strides: [1, 1]
      )

    <<a0::little-float-32, a1::little-float-32>> =
      arg1 |> Native.Memory.aligned() |> Native.OpaquePtr.to_binary(Integer.floor_div(32 * 2, 8))

    assert [a0, a1] == [1.100000023841858, 2.200000047683716]

    return =
      Native.Memory.new(
        nil,
        type: Native.F32,
        sizes: [1, 1],
        strides: [1, 1]
      )

    descriptor_str = return.descriptor |> Native.Memory.Descriptor.dump()

    assert descriptor_str =~
             "{ .allocated = null, .aligned = null, .offset = 0, .sizes = { 1, 1 }, .strides = { 1, 1 } }"

    ptr_str = Native.Memory.descriptor_ptr(return) |> Native.dump()
    # if the return is a struct, it should be first jit arg
    MLIR.ExecutionEngine.invoke!(
      jit,
      "test_multi_broadcast",
      Enum.map([return, arg0, arg1], &Native.Memory.descriptor_ptr/1)
    )

    # after invoke, the descriptor should be updated at the same address
    assert ptr_str == Native.Memory.descriptor_ptr(return) |> Native.dump()

    assert return.descriptor |> Native.Memory.Descriptor.dump() =~
             ".offset = 0, .sizes = { 2, 3 }, .strides = { 3, 1 }"

    assert return.descriptor |> Native.Memory.Descriptor.offset() == 0

    arg0
    |> Native.Memory.aligned()
    |> Native.OpaquePtr.to_binary(Integer.floor_div(32 * 3, 8))

    <<
      a0::little-float-32,
      a1::little-float-32
    >> =
      arg1
      |> Native.Memory.aligned()
      |> Native.OpaquePtr.to_binary(Integer.floor_div(32 * 2, 8))

    assert [a0, a1] == [1.100000023841858, 2.200000047683716]

    <<
      x0::little-float-32,
      x1::little-float-32,
      x2::little-float-32,
      x3::little-float-32,
      x4::little-float-32,
      x5::little-float-32
    >> =
      return
      # must use aligned ptr if it is allocated by LLVM
      |> Native.Memory.aligned()
      |> Native.OpaquePtr.to_binary(Integer.floor_div(32 * 6, 8))

    assert [x0, x1, x2, x3, x4, x5] == [
             2.4200000762939453,
             3.630000352859497,
             4.840000152587891,
             7.260000705718994,
             9.680000305175781,
             12.100000381469727
           ]

    assert :ok == Native.Memory.deallocate(return)
  end
end
