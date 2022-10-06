defmodule PDLTest do
  use ExUnit.Case
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  import Beaver.MLIR.Transforms

  @moduletag :pdl
  setup do
    [ctx: MLIR.Context.create(allow_unregistered: true)]
  end

  @apply_rewrite_op_patterns """
  module @patterns {
    pdl_interp.func @matcher(%root : !pdl.operation) {
      %test_attr = pdl_interp.create_attribute unit
      %attr = pdl_interp.get_attribute "test_attr" of %root
      pdl_interp.are_equal %test_attr, %attr : !pdl.attribute -> ^pat, ^end

    ^pat:
      pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

    ^end:
      pdl_interp.finalize
    }

    module @rewriters {
      pdl_interp.func @success(%root : !pdl.operation) {
        %op = pdl_interp.create_operation "test.success1"
        pdl_interp.erase %root
        pdl_interp.finalize
      }
    }
  }
  """

  @apply_rewrite_op_ir """
  module @ir attributes { test.are_equal_1 } {
    "test.op"() { test_attr } : () -> ()
  }
  """

  test "AreEqualOp", context do
    ctx = context[:ctx]
    CAPI.mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create(ctx, @apply_rewrite_op_patterns)

    inspector = fn
      {:successor, %CAPI.MlirBlock{} = successor}, acc ->
        {{:successor, successor}, acc}

      {:argument, %MLIR.Value{}} = argument, acc ->
        {argument, acc}

      {:result, %MLIR.Value{}} = result, acc ->
        {result, acc}

      {:operand, %MLIR.Value{}} = operand, acc ->
        {operand, acc}

      {name, %CAPI.MlirAttribute{} = attribute}, acc ->
        {{name, attribute}, acc}

      %CAPI.MlirOperation{} = mlir, acc ->
        mlir = %op{} = mlir |> Beaver.concrete()
        {mlir, [op | acc]}

      %element{} = mlir, acc ->
        {mlir, [element | acc]}
    end

    {mlir, acc} =
      pattern_module
      |> Beaver.Walker.traverse([], inspector, inspector)

    assert acc == [
             Beaver.MLIR.Dialect.Builtin.Module,
             Beaver.MLIR.CAPI.MlirRegion,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.Dialect.Builtin.Module,
             Beaver.MLIR.CAPI.MlirRegion,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.Dialect.PDLInterp.Func,
             Beaver.MLIR.CAPI.MlirRegion,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.Dialect.PDLInterp.Finalize,
             Beaver.MLIR.Dialect.PDLInterp.Finalize,
             Beaver.MLIR.Dialect.PDLInterp.Erase,
             Beaver.MLIR.Dialect.PDLInterp.Erase,
             Beaver.MLIR.Dialect.PDLInterp.CreateOperation,
             Beaver.MLIR.Dialect.PDLInterp.CreateOperation,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.CAPI.MlirRegion,
             Beaver.MLIR.Dialect.PDLInterp.Func,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.CAPI.MlirRegion,
             Beaver.MLIR.Dialect.Builtin.Module,
             Beaver.MLIR.Dialect.PDLInterp.Func,
             Beaver.MLIR.CAPI.MlirRegion,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.Dialect.PDLInterp.Finalize,
             Beaver.MLIR.Dialect.PDLInterp.Finalize,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.Dialect.PDLInterp.RecordMatch,
             Beaver.MLIR.Dialect.PDLInterp.RecordMatch,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.Dialect.PDLInterp.AreEqual,
             Beaver.MLIR.Dialect.PDLInterp.AreEqual,
             Beaver.MLIR.Dialect.PDLInterp.GetAttribute,
             Beaver.MLIR.Dialect.PDLInterp.GetAttribute,
             Beaver.MLIR.Dialect.PDLInterp.CreateAttribute,
             Beaver.MLIR.Dialect.PDLInterp.CreateAttribute,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.CAPI.MlirRegion,
             Beaver.MLIR.Dialect.PDLInterp.Func,
             Beaver.MLIR.CAPI.MlirBlock,
             Beaver.MLIR.CAPI.MlirRegion,
             Beaver.MLIR.Dialect.Builtin.Module
           ]

    assert mlir
           |> Beaver.Walker.container()
           |> MLIR.CAPI.mlirOperationEqual(MLIR.Operation.from_module(pattern_module))
           |> Beaver.Native.to_term()

    ir_module = MLIR.Module.create(ctx, @apply_rewrite_op_ir)
    MLIR.Operation.verify!(pattern_module)
    MLIR.Operation.verify!(ir_module)
    pdl_pattern = CAPI.beaverPDLPatternGet(pattern_module)
    pattern_set = CAPI.beaverRewritePatternSetGet(ctx)
    pattern_set = CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pdl_pattern)

    region =
      ir_module
      |> MLIR.Operation.from_module()
      |> CAPI.mlirOperationGetFirstRegion()

    result = CAPI.beaverApplyOwnedPatternSetOnRegion(region, pattern_set)

    assert Beaver.MLIR.LogicalResult.success?(result)

    ir_string = MLIR.to_string(ir_module)
    assert not String.contains?(ir_string, "test.op")
    assert String.contains?(ir_string, "test.success")
    CAPI.mlirContextDestroy(ctx)
  end

  # TODO: figure out why custom asm format of pdl doesn't work
  @are_equal_op_pdl Path.join(__DIR__, "pdl_erase_and_create.mlir") |> File.read!()

  test "AreEqualOp pdl version" do
    ctx = MLIR.Context.create()
    CAPI.mlirContextSetAllowUnregisteredDialects(ctx, true)
    pattern_module = MLIR.Module.create(ctx, @are_equal_op_pdl)
    assert not MLIR.Module.is_null(pattern_module), "fail to parse module"
    ir_module = MLIR.Module.create(ctx, @apply_rewrite_op_ir)
    MLIR.Operation.verify!(pattern_module)
    MLIR.Operation.verify!(ir_module)
    pattern_string = MLIR.to_string(pattern_module)
    assert String.contains?(pattern_string, "test.op")
    assert String.contains?(pattern_string, "test.success2")
    pdl_pattern = CAPI.beaverPDLPatternGet(pattern_module)
    pattern_set = CAPI.beaverRewritePatternSetGet(ctx)
    pattern_set = CAPI.beaverPatternSetAddOwnedPDLPattern(pattern_set, pdl_pattern)
    region = ir_module |> MLIR.Operation.from_module() |> CAPI.mlirOperationGetFirstRegion()
    result = CAPI.beaverApplyOwnedPatternSetOnRegion(region, pattern_set)

    assert MLIR.LogicalResult.success?(result), "fail to apply pattern"

    ir_string = MLIR.to_string(ir_module)
    assert not String.contains?(ir_string, "test.op")
    assert String.contains?(ir_string, "test.success2")
    CAPI.mlirContextDestroy(ctx)
  end

  test "load from string", context do
    %MLIR.CAPI.MlirPDLPatternModule{} =
      """
      module @erase {
        pdl.pattern : benefit(1) {
          %root = operation "foo.op"
          rewrite %root {
            erase %root
          }
        }
      }
      """
      |> MLIR.Pattern.from_string(ctx: context[:ctx])
  end

  test "replace tosa", context do
    defmodule TestTOSAPatterns do
      def gen_ir_module(ctx) do
        mlir ctx: ctx do
          module do
            Func.func test_multi_broadcast(
                        function_type:
                          Type.function(
                            [
                              Type.ranked_tensor([1, 3], Type.f32()),
                              Type.ranked_tensor([2, 1], Type.f32())
                            ],
                            [Type.ranked_tensor([2, 3], Type.f32())]
                          )
                      ) do
              region do
                block entry(
                        a >>> Type.ranked_tensor([1, 3], Type.f32()),
                        b >>> Type.ranked_tensor([2, 1], Type.f32())
                      ) do
                  res =
                    TOSA.add(a, b, one: MLIR.Attribute.integer(MLIR.Type.i32(), 1)) >>>
                      Type.ranked_tensor([2, 3], Type.f32())

                  res1 = TOSA.add(res, b) >>> Type.ranked_tensor([2, 3], Type.f32())
                  Func.return(res1) >>> []
                end
              end
            end
          end
        end
      end

      defpat replace_add_op(_t = %TOSA.Add{operands: [a, b], results: [res]}) do
        %MLIR.Value{} = res
        # create a common struct to see if the pattern transformation would skip it
        assert %Range{first: 1, last: 10, step: 2} |> Range.size() == 5
        %TOSA.Sub{operands: [a, b]}
      end

      defpat replace_multi_add_op(
               one = Attribute.integer(MLIR.Type.i32(), 1),
               _x = %Range{first: 1, last: 10, step: 2},
               ty = Type.ranked_tensor([2, 3], Type.f32()),
               %TOSA.Add{
                 operands: [a, b],
                 results: [res],
                 attributes: [one: ^one]
               },
               _t = %TOSA.Add{
                 operands: [
                   ^res,
                   ^b
                 ],
                 results: [^ty]
               }
             ) do
        alias Beaver.DSL.Pattern
        types = [Type.ranked_tensor([2, 3], Type.f32())]
        env = %Pattern.Env{ctx: MLIR.__CONTEXT__(), block: MLIR.__BLOCK__()}
        a = %TOSA.Sub{operands: [a, b], results: types}
        a = Pattern.result(env, a, 0)
        a = %TOSA.Sub{operands: [a, b], results: types}
        a = Pattern.result(env, a, 0)
        a = %TOSA.Sub{operands: [a, b], results: types}
        a = Pattern.result(env, a, 0)
        a = %TOSA.Sub{operands: [a, b], results: types, attributes: [one: one]}
        a = Pattern.result(env, a, 0)
        %TOSA.Sub{operands: [a, b]}
      end

      defpat replace_multi_add_op1(
               one = Attribute.integer(MLIR.Type.i32(), 1),
               _ty = Type.ranked_tensor([2, 3], Type.f32()),
               %TOSA.Add{
                 operands: [a, b],
                 results: [res],
                 attributes: [one: ^one]
               },
               _t = %TOSA.Add{
                 operands: [
                   ^res,
                   ^b
                 ],
                 results: [ty]
               }
             ) do
        %MLIR.Value{} = ty
        %TOSA.Sub{operands: [a, b]}
      end

      defpat replace_multi_add_op2(
               one = Attribute.integer(MLIR.Type.i32(), 1),
               types = [Type.ranked_tensor([2, 3], Type.f32())],
               %TOSA.Add{
                 operands: [a, b],
                 results: [res],
                 attributes: [one: ^one]
               },
               _t = %TOSA.Add{
                 operands: [
                   ^res,
                   ^b
                 ],
                 results: ^types
               }
             ) do
        %TOSA.Sub{operands: [a, b]}
      end

      defpat replace_multi_add_op3(
               _one = Attribute.integer(MLIR.Type.i32(), 1),
               types = [Type.ranked_tensor([2, 3], Type.f32())],
               %TOSA.Add{
                 operands: [a, b],
                 results: [res],
                 attributes: [one: one]
               },
               _t = %TOSA.Add{
                 operands: [
                   ^res,
                   ^b
                 ],
                 results: types
               }
             ) do
        %MLIR.Value{} = one
        %TOSA.Sub{operands: [a, b]}
      end
    end

    ctx = context[:ctx]
    opts = [ctx: ctx]

    for pattern <- [
          TestTOSAPatterns.replace_add_op(),
          TestTOSAPatterns.replace_multi_add_op(opts),
          TestTOSAPatterns.replace_multi_add_op1(opts),
          TestTOSAPatterns.replace_multi_add_op2(opts),
          TestTOSAPatterns.replace_multi_add_op3(opts)
        ] do
      ir_module = TestTOSAPatterns.gen_ir_module(ctx)
      MLIR.Operation.verify!(ir_module)
      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "tosa.sub"), ir_string

      MLIR.Pattern.apply!(ir_module, [
        pattern
      ])
      |> MLIR.Operation.verify!(dump_if_fail: true)
      |> MLIR.Transforms.canonicalize()
      |> MLIR.Pass.Composer.run!()

      ir_string = MLIR.to_string(ir_module)
      assert not String.contains?(ir_string, "tosa.add"), ir_string
      assert String.contains?(ir_string, "tosa.sub"), ir_string
    end
  end

  test "toy compiler with pass", context do
    alias Beaver.MLIR.Dialect.Func

    ctx = context[:ctx]

    defmodule ToyPass do
      use Beaver.MLIR.Pass, on: Func.Func

      defpat replace_add_op(_t = %TOSA.Add{operands: [a, b], results: [res], attributes: []}) do
        %MLIR.Value{} = res
        %TOSA.Sub{operands: [a, b]}
      end

      def run(%MLIR.CAPI.MlirOperation{} = operation) do
        with %Func.Func{attributes: attributes} <- Beaver.concrete(operation),
             2 <- Enum.count(attributes),
             {:ok, _} <- MLIR.Pattern.apply_(operation, [replace_add_op(benefit: 2)]) do
          :ok
        end
      end
    end

    ir =
      ~m"""
      module {
        func.func @tosa_add(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
          %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
          return %0 : tensor<2x3xf32>
        }
      }
      """.(ctx)
      |> MLIR.Pass.Composer.nested(Func.Func, [
        ToyPass.create()
      ])
      |> canonicalize
      |> MLIR.Pass.Composer.run!()

    ir_string = MLIR.to_string(ir)
    assert not (ir_string =~ "tosa.add")
    assert ir_string =~ "tosa.sub"
  end

  test "raise" do
    assert_raise RuntimeError,
                 ~r"Must pass a list or a variable to operands/attributes/results, got: \n%{a: 1}",
                 fn ->
                   defmodule PassWithIllegalPattern do
                     use Beaver

                     defpat replace_add_op(
                              _t = %TOSA.Add{operands: %{a: 1}, results: [_res], attributes: []}
                            ) do
                       %TOSA.Sub{operands: [a, b]}
                     end
                   end
                 end
  end
end