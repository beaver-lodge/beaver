defmodule MlirTest do
  use Beaver.Case, async: true

  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  @moduletag :smoke
  test "call wrapped apis" do
    ctx = MLIR.Context.create()

    location =
      MLIR.CAPI.mlirLocationFileLineColGet(
        ctx,
        MLIR.StringRef.create(__DIR__),
        1,
        2
      )

    _module = MLIR.CAPI.mlirModuleCreateEmpty(location)

    module =
      MLIR.Module.create(ctx, """
      func.func private @printNewline()
      func.func private @printI64(i64)
      """)

    MLIR.Operation.verify!(module)
    _operation = MLIR.CAPI.mlirModuleGetOperation(module)

    _ret_str = MLIR.StringRef.create("func.return")

    operation_state = %MLIR.Operation.State{name: "func.return", location: location}

    for _i <- 0..200 do
      operation_state_ptr =
        operation_state |> MLIR.Operation.State.create() |> Beaver.Native.ptr()

      _ret_op = MLIR.CAPI.mlirOperationCreate(operation_state_ptr)
    end

    i64_t = MLIR.CAPI.mlirTypeParseGet(ctx, MLIR.StringRef.create("i64"))
    # create func body entry block
    funcBodyArgTypes = [i64_t]
    funcBodyArgLocs = [location]
    funcBodyRegion = MLIR.CAPI.mlirRegionCreate()
    funcBody = MLIR.Block.create(funcBodyArgTypes, funcBodyArgLocs)
    [arg1] = funcBody |> MLIR.Block.add_arg!(ctx, ["i64"])
    # append block to region
    MLIR.CAPI.mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody)
    # create func
    operation_state =
      %MLIR.Operation.State{name: "func.func", context: ctx}
      |> MLIR.Operation.State.add_argument(
        sym_name: "\"add\"",
        function_type: "(i64, i64) -> (i64)"
      )
      |> MLIR.Operation.State.add_argument(funcBodyRegion)

    func_op = operation_state |> MLIR.Operation.create()

    add_op_state =
      %MLIR.Operation.State{name: "arith.addi", location: location}
      |> MLIR.Operation.State.add_argument(MLIR.Block.get_arg!(funcBody, 0))
      |> MLIR.Operation.State.add_argument(arg1)
      |> MLIR.Operation.State.add_argument({:result_types, ["i64"]})
      |> MLIR.Operation.State.create()

    name = MLIR.CAPI.beaverMlirOperationStateGetName(add_op_state)

    assert 10 == MLIR.CAPI.beaverStringRefGetLength(name) |> Beaver.Native.to_term()

    location1 = add_op_state |> MLIR.CAPI.beaverMlirOperationStateGetLocation()
    nResults = add_op_state |> MLIR.CAPI.beaverMlirOperationStateGetNumResults()
    nOperands = add_op_state |> MLIR.CAPI.beaverMlirOperationStateGetNumOperands()
    nRegions = add_op_state |> MLIR.CAPI.beaverMlirOperationStateGetNumRegions()
    nAttributes = add_op_state |> MLIR.CAPI.beaverMlirOperationStateGetNumAttributes()

    assert 0 == nRegions |> Beaver.Native.to_term()
    assert 1 == nResults |> Beaver.Native.to_term()
    assert 2 == nOperands |> Beaver.Native.to_term()
    assert 0 == nAttributes |> Beaver.Native.to_term()
    add_op = add_op_state |> MLIR.Operation.create()

    _ctx = MLIR.CAPI.mlirLocationGetContext(location)
    ctx = MLIR.CAPI.mlirLocationGetContext(location1)

    r = MLIR.CAPI.mlirOperationGetResult(add_op, 0)

    return_op =
      %MLIR.Operation.State{name: "func.return", context: ctx}
      |> MLIR.Operation.State.add_argument(r)
      |> MLIR.Operation.create()

    MLIR.CAPI.mlirBlockInsertOwnedOperation(funcBody, 0, add_op)
    MLIR.CAPI.mlirBlockInsertOwnedOperationAfter(funcBody, add_op, return_op)
    moduleBody = MLIR.CAPI.mlirModuleGetBody(module)
    MLIR.CAPI.mlirBlockInsertOwnedOperation(moduleBody, 0, func_op)

    MLIR.Operation.verify!(module, dump_if_fail: true)

    MLIR.CAPI.mlirContextDestroy(ctx)
  end

  test "elixir dialect" do
    require MLIR.Context

    ctx = MLIR.Context.create()

    # This api might trigger NDEBUG assert, so run it more
    for _ <- 1..200 do
      Task.async(fn ->
        :ok =
          MLIR.CAPI.mlirDialectHandleRegisterDialect(
            MLIR.CAPI.mlirGetDialectHandle__elixir__(),
            ctx
          )
      end)
    end
    |> Task.await_many()

    MLIR.CAPI.mlirContextLoadAllAvailableDialects(ctx)

    _add_op =
      %MLIR.Operation.State{name: "elixir.add", context: ctx}
      |> MLIR.Operation.create()
  end

  alias Beaver.MLIR.CAPI

  def create_adder_module(ctx) do
    MLIR.Module.create(ctx, """
    func.func @foo(%arg0 : i32) -> i32 {
      %res = arith.addi %arg0, %arg0 : i32
      return %res : i32
    }
    """)
  end

  def create_redundant_transpose_module(ctx) do
    MLIR.Module.create(ctx, """
    func.func @test_transpose(%arg0: tensor<1x2x3xi32>) -> () {
      %0 = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
      %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x2x3xi32>, tensor<3xi32>) -> (tensor<2x3x1xi32>)
      %2 = arith.constant dense<[2, 0, 1]> : tensor<3xi32>
      %3 = "tosa.transpose"(%1, %2) : (tensor<2x3x1xi32>, tensor<3xi32>) -> (tensor<1x2x3xi32>)
      return
    }
    """)
  end

  test "run a simple pass" do
    ctx = MLIR.Context.create()

    module = create_adder_module(ctx)
    module_op = module |> MLIR.CAPI.mlirModuleGetOperation()

    assert MLIR.CAPI.mlirOperationVerify(module_op) |> Beaver.Native.to_term()
    pm = CAPI.mlirPassManagerCreate(ctx)
    CAPI.mlirPassManagerAddOwnedPass(pm, CAPI.mlirCreateTransformsCSE())
    success = CAPI.mlirPassManagerRun(pm, module)

    assert success
           |> CAPI.beaverLogicalResultIsSuccess()
           |> Beaver.Native.to_term()

    CAPI.mlirPassManagerDestroy(pm)
    CAPI.mlirModuleDestroy(module)
    CAPI.mlirContextDestroy(ctx)
  end

  defmodule TransposeOp do
    defstruct a: nil, b: nil
  end

  defmodule TestPass do
    def run(%Beaver.MLIR.CAPI.MlirOperation{} = op) do
      MLIR.Operation.verify!(op)
      :ok
    end
  end

  test "Run a generic pass" do
    ctx = MLIR.Context.create()
    module = create_adder_module(ctx)
    assert not MLIR.Module.is_null(module)
    # TODO: create a supervisor to manage a TypeIDAllocator by mlir application
    typeIDAllocator = CAPI.mlirTypeIDAllocatorCreate()

    external = %MLIR.Pass{} = MLIR.ExternalPass.create(TestPass, "")

    pm = CAPI.mlirPassManagerCreate(ctx)
    CAPI.mlirPassManagerAddOwnedPass(pm, external)
    CAPI.mlirPassManagerAddOwnedPass(pm, CAPI.mlirCreateTransformsCSE())
    success = CAPI.mlirPassManagerRun(pm, module)

    assert Beaver.MLIR.LogicalResult.success?(success)

    CAPI.mlirPassManagerDestroy(pm)
    CAPI.mlirModuleDestroy(module)
    CAPI.mlirTypeIDAllocatorDestroy(typeIDAllocator)
    CAPI.mlirContextDestroy(ctx)
    # TODO: values above could be moved to setup
  end

  test "Run a func operation pass" do
    ctx = MLIR.Context.create()
    module = create_adder_module(ctx)
    assert not MLIR.Module.is_null(module)
    # TODO: create a supervisor to manage a TypeIDAllocator by mlir application

    external = %MLIR.Pass{} = MLIR.ExternalPass.create(TestPass, "func.func")

    pm = CAPI.mlirPassManagerCreate(ctx)
    npm = CAPI.mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create("func.func"))
    CAPI.mlirOpPassManagerAddOwnedPass(npm, external)
    success = CAPI.mlirPassManagerRun(pm, module)

    # equivalent to mlirLogicalResultIsSuccess
    assert Beaver.MLIR.LogicalResult.success?(success)

    CAPI.mlirPassManagerDestroy(pm)
    CAPI.mlirModuleDestroy(module)
    CAPI.mlirContextDestroy(ctx)
    # TODO: values above could be moved to setup
  end

  test "Run pass with patterns" do
    ctx = MLIR.Context.create()
    module = create_redundant_transpose_module(ctx)
    assert not MLIR.Module.is_null(module)
    external = %MLIR.Pass{} = MLIR.ExternalPass.create(TestPass, "func.func")

    pm = CAPI.mlirPassManagerCreate(ctx)
    npm = CAPI.mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create("func.func"))
    CAPI.mlirOpPassManagerAddOwnedPass(npm, external)
    success = CAPI.mlirPassManagerRun(pm, module)
    assert Beaver.MLIR.LogicalResult.success?(success)

    CAPI.mlirPassManagerDestroy(pm)
    CAPI.mlirModuleDestroy(module)
    CAPI.mlirContextDestroy(ctx)
    # TODO: values above could be moved to setup
  end

  defmacro some_constrain(_t) do
    true
  end

  defmacro has_one_use(_v) do
    true
  end

  import Beaver.MLIR.CAPI

  def lower_to_llvm(ctx, module) do
    pm = mlirPassManagerCreate(ctx)

    opm =
      mlirPassManagerGetNestedUnder(
        pm,
        MLIR.StringRef.create("func.func")
      )

    mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVM())

    mlirOpPassManagerAddOwnedPass(
      opm,
      mlirCreateTransformsPrintOpStats()
    )

    status = mlirPassManagerRun(pm, module)

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure running pass pipeline"
    end

    mlirPassManagerDestroy(pm)
    module
  end

  test "basic jit" do
    module_str = """
    module {
      func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
        %res = arith.addi %arg0, %arg0 : i32
        return %res : i32
      }
    }
    """

    ctx = MLIR.Context.create()
    module = MLIR.Module.create(ctx, module_str)
    MLIR.Operation.verify!(module)
    lower_to_llvm(ctx, module)
    jit = MLIR.ExecutionEngine.create!(module)
    arg = Beaver.Native.I32.make(42)
    return = Beaver.Native.I32.make(-1)

    before_ptr = return |> Beaver.Native.opaque_ptr() |> Beaver.Native.to_term()
    return = MLIR.ExecutionEngine.invoke!(jit, "add", [arg], return)
    after_ptr = return |> Beaver.Native.opaque_ptr() |> Beaver.Native.to_term()

    assert before_ptr == after_ptr

    return |> Beaver.Native.opaque_ptr() |> Beaver.Native.to_term()
    assert return |> Beaver.Native.to_term() == 84

    for i <- 0..100_0 do
      Task.async(fn ->
        arg = Beaver.Native.I32.make(i)
        return = Beaver.Native.I32.make(-1)
        return = MLIR.ExecutionEngine.invoke!(jit, "add", [arg], return)
        # return here is a resource reference
        assert return == return
        assert return |> Beaver.Native.to_term() == i * 2
      end)
    end
    |> Task.await_many()

    MLIR.ExecutionEngine.destroy(jit)
    MLIR.Module.destroy(module)
    CAPI.mlirContextDestroy(ctx)
  end

  test "affine expr and map", context do
    ctx = context[:ctx]
    affineDimExpr = MLIR.CAPI.mlirAffineDimExprGet(ctx, 0)
    affineSymbolExpr = MLIR.CAPI.mlirAffineSymbolExprGet(ctx, 1)
    exprs = MLIR.CAPI.MlirAffineExpr.array([affineDimExpr, affineSymbolExpr], mut: true)
    map = MLIR.CAPI.mlirAffineMapGet(ctx, 3, 3, 2, exprs)
    txt = "(d0, d1, d2)[s0, s1, s2] -> (d0, s1)"
    assert map |> MLIR.to_string() == txt
    assert MLIR.Attribute.affine_map(map) |> MLIR.to_string() == "affine_map<#{txt}>"

    assert MLIR.AffineMap.create(3, 3, [MLIR.AffineMap.dim(0), MLIR.AffineMap.symbol(1)]).(ctx)
           |> MLIR.to_string() == txt
  end
end
