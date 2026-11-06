defmodule StructuredComputeTest do
  use Beaver.Case, async: true
  use Beaver

  alias Beaver.MLIR
  alias Beaver.MLIR.{AffineMap, Attribute, Type}
  alias Beaver.MLIR.Dialect.{Arith, Func, Linalg, Tensor, Vector}

  require Func
  require Linalg

  @moduletag :smoke

  test "builds tensor slices from mixed static and dynamic entries", %{ctx: ctx} do
    module =
      mlir ctx: ctx do
        module do
          Func.func slice(function_type: Type.function([~t{tensor<8xf32>}], [~t{tensor<4xf32>}])) do
            region do
              block _entry(source >>> ~t{tensor<8xf32>}) do
                result =
                  Tensor.extract_slice(
                    source: source,
                    offsets: [],
                    sizes: [],
                    strides: [],
                    static_offsets: Attribute.dense_array([2], Beaver.Native.I64),
                    static_sizes: Attribute.dense_array([4], Beaver.Native.I64),
                    static_strides: Attribute.dense_array([1], Beaver.Native.I64),
                    operand_segment_sizes: :infer
                  ) >>>
                    ~t{tensor<4xf32>}

                Func.return(result) >>> []
              end
            end
          end
        end
      end

    MLIR.verify!(module)
    assert to_string(module) =~ "tensor.extract_slice %arg0[2] [4] [1]"
  end

  test "builds vector transfer reads with structured attributes", %{ctx: ctx} do
    map = AffineMap.create(1, 0, [AffineMap.dim(0)])

    module =
      mlir ctx: ctx do
        module do
          Func.func read(function_type: Type.function([~t{memref<8xf32>}], [])) do
            region do
              block _entry(source >>> ~t{memref<8xf32>}) do
                index = Arith.constant(value: Attribute.index(0)) >>> Type.index()
                padding = Arith.constant(value: Attribute.float(Type.f32(), 0.0)) >>> Type.f32()

                _vector =
                  Vector.transfer_read(
                    base: source,
                    padding: padding,
                    indices: [index],
                    mask: [],
                    permutation_map: Attribute.affine_map(map),
                    in_bounds: Attribute.array([Attribute.bool(true)]),
                    operand_segment_sizes: :infer
                  ) >>> ~t{vector<4xf32>}

                Func.return() >>> []
              end
            end
          end
        end
      end

    MLIR.verify!(module)
    assert to_string(module) =~ "vector.transfer_read"
    assert to_string(module) =~ "in_bounds = [true]"
  end

  test "builds linalg.generic and lowers it through bufferization to LLVM", %{ctx: ctx} do
    tensor_type = ~t{tensor<4xf32>}
    identity = AffineMap.create(1, 0, [AffineMap.dim(0)])

    module =
      mlir ctx: ctx do
        module do
          Func.func add(function_type: Type.function([tensor_type, tensor_type], [tensor_type])) do
            region do
              block _entry(lhs >>> tensor_type, rhs >>> tensor_type) do
                init = Tensor.empty() >>> tensor_type

                result =
                  Linalg.generic inputs: [lhs, rhs],
                                 outputs: [init],
                                 indexing_maps: [identity, identity, identity],
                                 iterators: [:parallel] do
                    fn [left, right], [_out] -> Arith.addf(left, right) >>> Type.f32() end
                  end

                Func.return(result) >>> []
              end
            end
          end
        end
      end

    MLIR.verify!(module)
    assert to_string(module) =~ "linalg.generic"

    pipeline =
      "one-shot-bufferize{bufferize-function-boundaries=true unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map},convert-linalg-to-loops,canonicalize,lower-affine,convert-scf-to-cf,convert-index-to-llvm,convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts"

    assert ^module = module |> Beaver.Composer.append(pipeline) |> Beaver.Composer.run!()
    MLIR.verify!(module)
    assert to_string(module) =~ "llvm.func @add"
    refute to_string(module) =~ "linalg.generic"
    refute to_string(module) =~ "tensor.empty"
  end
end
