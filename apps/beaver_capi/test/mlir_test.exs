defmodule MlirTest do
  use ExUnit.Case, async: true
  use Beaver
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  test "call wrapped apis" do
    ctx =
      MLIR.CAPI.mlirContextCreate()
      |> Exotic.Value.transmit()

    MLIR.CAPI.mlirRegisterAllDialects(ctx)

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

    Exotic.LibC.load()

    _ret_str = MLIR.StringRef.create("func.return")

    operation_state = MLIR.Operation.State.get!("func.return", location)

    for _i <- 0..200 do
      operation_state_ptr = operation_state |> Exotic.Value.get_ptr()
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
    operation_state = ctx |> MLIR.Operation.State.get!("func.func")

    MLIR.Operation.State.add_attr(operation_state,
      sym_name: "\"add\"",
      function_type: "(i64, i64) -> (i64)"
    )

    MLIR.Operation.State.add_regions(operation_state, [funcBodyRegion])
    func_op = operation_state |> MLIR.Operation.create()

    add_op_state =
      MLIR.Operation.State.get!("arith.addi", location)
      |> MLIR.Operation.State.add_operand([MLIR.Block.get_arg!(funcBody, 0), arg1])
      |> MLIR.Operation.State.add_result(["i64"])

    name = Exotic.Value.fetch(add_op_state, MLIR.CAPI.MlirOperationState, :name)

    assert 10 ==
             Exotic.Value.fetch(name, MLIR.CAPI.MlirStringRef, :length)
             |> Exotic.Value.extract()

    location1 = Exotic.Value.fetch(add_op_state, MLIR.CAPI.MlirOperationState, :location)
    nResults = Exotic.Value.fetch(add_op_state, MLIR.CAPI.MlirOperationState, :nResults)
    nOperands = Exotic.Value.fetch(add_op_state, MLIR.CAPI.MlirOperationState, :nOperands)
    nRegions = Exotic.Value.fetch(add_op_state, MLIR.CAPI.MlirOperationState, :nRegions)
    nAttributes = Exotic.Value.fetch(add_op_state, MLIR.CAPI.MlirOperationState, :nAttributes)

    assert 0 == nRegions |> Exotic.Value.extract()
    assert 1 == nResults |> Exotic.Value.extract()
    assert 2 == nOperands |> Exotic.Value.extract()
    assert 0 == nAttributes |> Exotic.Value.extract()
    add_op = add_op_state |> MLIR.Operation.create()

    _ctx = MLIR.CAPI.mlirLocationGetContext(location)
    ctx = MLIR.CAPI.mlirLocationGetContext(location1)

    r = MLIR.CAPI.mlirOperationGetResult(add_op, 0)

    return_op =
      ctx
      |> MLIR.Operation.State.get!("func.return")
      |> MLIR.Operation.State.add_operand([r])
      |> MLIR.Operation.create()

    MLIR.CAPI.mlirBlockInsertOwnedOperation(funcBody, 0, add_op)
    MLIR.CAPI.mlirBlockInsertOwnedOperationAfter(funcBody, add_op, return_op)
    moduleBody = MLIR.CAPI.mlirModuleGetBody(module)
    MLIR.CAPI.mlirBlockInsertOwnedOperation(moduleBody, 0, func_op)

    MLIR.Operation.verify!(module)

    MLIR.CAPI.mlirContextDestroy(ctx)
  end

  test "elixir dialect" do
    require MLIR.Context

    ctx =
      MLIR.CAPI.mlirContextCreate()
      |> Exotic.Value.transmit()

    # This api might trigger NDEBUG assert, so run it more
    for _ <- 1..200 do
      Task.async(fn ->
        %Exotic.Value{
          type: :void
        } =
          MLIR.CAPI.mlirDialectHandleRegisterDialect(
            MLIR.CAPI.mlirGetDialectHandle__elixir__(),
            ctx
          )
      end)
    end
    |> Task.await_many()

    MLIR.CAPI.mlirRegisterAllDialects(ctx)

    _add_op =
      MLIR.Operation.State.get!(ctx, "elixir.add")
      |> MLIR.Operation.State.add_operand([])
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
    ctx =
      MLIR.CAPI.mlirContextCreate()
      |> Exotic.Value.transmit()

    CAPI.mlirRegisterAllDialects(ctx)

    module = create_adder_module(ctx)

    assert MLIR.CAPI.mlirOperationVerify(module) |> Exotic.Value.extract()

    CAPI.MlirExternalPass.native_fields_with_names()

    CAPI.MlirExternalPassCallbacks.native_fields_with_names()

    CAPI.MlirStringCallback.module_info()[:attributes]

    pm = CAPI.mlirPassManagerCreate(ctx)
    CAPI.mlirPassManagerAddOwnedPass(pm, CAPI.mlirCreateTransformsCSE())
    success = CAPI.mlirPassManagerRun(pm, module)

    # equivalent to mlirLogicalResultIsSuccess
    # TODO: add a Exotic.Value.as_bool/1
    assert success
           |> Exotic.Value.fetch(MLIR.CAPI.MlirLogicalResult, :value)
           |> Exotic.Value.extract() != 0

    CAPI.mlirPassManagerDestroy(pm)
    CAPI.mlirModuleDestroy(module)
    CAPI.mlirContextDestroy(ctx)
  end

  defmodule TransposeOp do
    defstruct a: nil, b: nil
  end

  defmodule TestPass do
    # use MLIR.Pass
    @behaviour MLIR.CAPI.MlirExternalPassCallbacks.Closures
    defmodule State do
      use MLIR.Pass.State
      defstruct init: false, construct: false, destruct: false, run: false, clone: false
    end

    def fold_transpose(a = %TransposeOp{a: %TransposeOp{}}) do
      a
    end

    @impl true
    def handle_invoke(:construct, [a], state) do
      {:return, a, state}
    end

    def handle_invoke(:destruct, [a], state) do
      {:return, a, state}
    end

    def handle_invoke(:initialize, [_ctx, userData], state) do
      {:return, userData, state}
    end

    def handle_invoke(:run, [op, _pass, userData], state) do
      MLIR.Operation.verify!(op)
      {:return, userData, state}
    end

    def handle_invoke(:clone, [_a], state) do
      {:pass, state}
    end
  end

  test "Run a generic pass" do
    ctx = MLIR.Context.create()
    module = create_adder_module(ctx)
    assert not MLIR.Module.is_null(module)
    # TODO: create a supervisor to manage a TypeIDAllocator by mlir application
    typeIDAllocator = CAPI.mlirTypeIDAllocatorCreate()

    %MLIR.Pass{external: external} =
      MLIR.Pass.create(TestPass, %TestPass.State{}, typeIDAllocator)

    pm = CAPI.mlirPassManagerCreate(ctx)
    CAPI.mlirPassManagerAddOwnedPass(pm, external)
    CAPI.mlirPassManagerAddOwnedPass(pm, CAPI.mlirCreateTransformsCSE())
    success = CAPI.mlirPassManagerRun(pm, module)

    # equivalent to mlirLogicalResultIsSuccess
    # TODO: add a Exotic.Value.as_bool/1
    assert success
           |> Exotic.Value.fetch(MLIR.CAPI.MlirLogicalResult, :value)
           |> Exotic.Value.extract() != 0

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
    typeIDAllocator = CAPI.mlirTypeIDAllocatorCreate()

    %MLIR.Pass{external: external} =
      MLIR.Pass.create(TestPass, %TestPass.State{}, typeIDAllocator, "func.func")

    pm = CAPI.mlirPassManagerCreate(ctx)
    npm = CAPI.mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create("func.func"))
    CAPI.mlirPassManagerAddOwnedPass(npm, external)
    success = CAPI.mlirPassManagerRun(pm, module)

    # equivalent to mlirLogicalResultIsSuccess
    # TODO: add a Exotic.Value.as_bool/1
    assert success
           |> Exotic.Value.fetch(MLIR.CAPI.MlirLogicalResult, :value)
           |> Exotic.Value.extract() != 0

    CAPI.mlirPassManagerDestroy(pm)
    CAPI.mlirModuleDestroy(module)
    CAPI.mlirTypeIDAllocatorDestroy(typeIDAllocator)
    CAPI.mlirContextDestroy(ctx)
    # TODO: values above could be moved to setup
  end

  test "Run pass with patterns" do
    ctx = MLIR.Context.create()
    module = create_redundant_transpose_module(ctx)
    assert not MLIR.Module.is_null(module)
    # TODO: create a supervisor to manage a TypeIDAllocator by mlir application
    typeIDAllocator = CAPI.mlirTypeIDAllocatorCreate()

    %MLIR.Pass{external: external} =
      MLIR.Pass.create(TestPass, %TestPass.State{}, typeIDAllocator, "func.func")

    pm = CAPI.mlirPassManagerCreate(ctx)
    npm = CAPI.mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create("func.func"))
    CAPI.mlirPassManagerAddOwnedPass(npm, external)
    success = CAPI.mlirPassManagerRun(pm, module)

    # equivalent to mlirLogicalResultIsSuccess
    # TODO: add a Exotic.Value.as_bool/1
    assert success
           |> Exotic.Value.fetch(MLIR.CAPI.MlirLogicalResult, :value)
           |> Exotic.Value.extract() != 0

    CAPI.mlirPassManagerDestroy(pm)
    CAPI.mlirModuleDestroy(module)
    CAPI.mlirTypeIDAllocatorDestroy(typeIDAllocator)
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
      mlirCreateConversionConvertArithmeticToLLVM()
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
    arg = Exotic.Value.get(42)
    return = Exotic.Value.get(-1)
    return = MLIR.ExecutionEngine.invoke!(jit, "add", [arg], return)
    assert return |> Exotic.Value.extract() == 84

    for i <- 0..100_0 do
      Task.async(fn ->
        arg = Exotic.Value.get(i)
        return = Exotic.Value.get(-1)
        return = MLIR.ExecutionEngine.invoke!(jit, "add", [arg], return)
        # return here is a resource reference
        assert return == return
        assert return |> Exotic.Value.extract() == i * 2
      end)
    end
    |> Task.await_many()

    MLIR.ExecutionEngine.destroy(jit)
    MLIR.Module.destroy(module)
    CAPI.mlirContextDestroy(ctx)
  end
end
