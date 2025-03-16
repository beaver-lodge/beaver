defmodule MlirTest do
  use Beaver.Case, async: true
  use Beaver
  alias Beaver.MLIR
  import Beaver.MLIR.CAPI

  @moduletag :smoke
  test "call wrapped apis" do
    ctx = MLIR.Context.create()

    location =
      mlirLocationFileLineColGet(
        ctx,
        MLIR.StringRef.create(__DIR__),
        1,
        2
      )

    _module = mlirModuleCreateEmpty(location)

    module =
      MLIR.Module.create!(
        """
        func.func private @printNewline()
        func.func private @printI64(i64)
        """,
        ctx: ctx
      )

    _operation = mlirModuleGetOperation(module)

    _ret_str = MLIR.StringRef.create("func.return")

    changeset = %Beaver.Changeset{name: "func.return", location: location}

    for _i <- 0..200 do
      changeset |> MLIR.Operation.State.create() |> Beaver.Native.ptr() |> mlirOperationCreate()
    end

    i64_t = mlirTypeParseGet(ctx, MLIR.StringRef.create("i64"))
    # create func body entry block
    func_body_arg_types = [i64_t]
    func_body_arg_locs = [location]
    func_body_region = mlirRegionCreate()
    func_body = MLIR.Block.create(func_body_arg_types, func_body_arg_locs)
    [arg1] = func_body |> MLIR.Block.add_args!(["i64"], ctx: ctx)
    # append block to region
    mlirRegionAppendOwnedBlock(func_body_region, func_body)
    # create func
    changeset =
      %Beaver.Changeset{name: "func.func", context: ctx}
      |> Beaver.Changeset.add_argument(
        sym_name: "\"add\"",
        function_type: "(i64, i64) -> (i64)"
      )
      |> Beaver.Changeset.add_argument(func_body_region)

    func_op = changeset |> MLIR.Operation.create()

    add_op_state =
      %Beaver.Changeset{name: "arith.addi", location: location}
      |> Beaver.Changeset.add_argument(MLIR.Block.get_arg!(func_body, 0))
      |> Beaver.Changeset.add_argument(arg1)
      |> Beaver.Changeset.add_argument({:result_types, ["i64"]})
      |> MLIR.Operation.State.create()

    name = beaverOperationStateGetName(add_op_state)

    assert 10 == beaverStringRefGetLength(name) |> Beaver.Native.to_term()

    location1 = add_op_state |> beaverOperationStateGetLocation()
    n_results = add_op_state |> beaverOperationStateGetNumResults()
    n_operands = add_op_state |> beaverOperationStateGetNumOperands()
    n_regions = add_op_state |> beaverOperationStateGetNumRegions()
    n_attributes = add_op_state |> beaverOperationStateGetNumAttributes()

    assert 0 == n_regions |> Beaver.Native.to_term()
    assert 1 == n_results |> Beaver.Native.to_term()
    assert 2 == n_operands |> Beaver.Native.to_term()
    assert 0 == n_attributes |> Beaver.Native.to_term()
    add_op = add_op_state |> MLIR.Operation.create()

    _ctx = MLIR.context(location)
    ctx = MLIR.context(location1)

    r = mlirOperationGetResult(add_op, 0)

    return_op =
      %Beaver.Changeset{name: "func.return", context: ctx}
      |> Beaver.Changeset.add_argument(r)
      |> MLIR.Operation.create()

    mlirBlockInsertOwnedOperation(func_body, 0, add_op)
    mlirBlockInsertOwnedOperationAfter(func_body, add_op, return_op)
    module_body = mlirModuleGetBody(module)
    mlirBlockInsertOwnedOperation(module_body, 0, func_op)
    MLIR.verify!(module)
    mlirContextDestroy(ctx)
  end

  test "elixir dialect" do
    require MLIR.Context

    ctx = MLIR.Context.create()

    # This api might trigger NDEBUG assert, so run it more
    for _ <- 1..200 do
      Task.async(fn ->
        :ok =
          mlirDialectHandleRegisterDialect(
            mlirGetDialectHandle__elixir__(),
            ctx
          )
      end)
    end
    |> Task.await_many()

    mlirContextLoadAllAvailableDialects(ctx)

    _add_op =
      %Beaver.Changeset{name: "elixir.add", context: ctx}
      |> MLIR.Operation.create()
  end

  def create_adder_module(ctx) do
    MLIR.Module.create(
      """
      func.func @foo(%arg0 : i32) -> i32 {
        %res = arith.addi %arg0, %arg0 : i32
        return %res : i32
      }
      """,
      ctx: ctx
    )
  end

  def create_redundant_transpose_module(ctx) do
    MLIR.Module.create("""
    func.func @test_transpose(%arg0: tensor<1x2x3xi32>) -> () {
      %1 = "tosa.transpose"(%arg0) {perms = array<i32: 1, 2, 0>} : (tensor<1x2x3xi32>) -> (tensor<2x3x1xi32>)
      %3 = "tosa.transpose"(%1) {perms = array<i32: 2, 0, 1>} : (tensor<2x3x1xi32>) -> (tensor<1x2x3xi32>)
      return
    }
    """).(ctx)
  end

  test "run a simple pass" do
    ctx = MLIR.Context.create()

    {:ok, module} = create_adder_module(ctx)
    module_op = module |> mlirModuleGetOperation()

    assert mlirOperationVerify(module_op) |> Beaver.Native.to_term()
    pm = mlirPassManagerCreate(ctx)
    mlirPassManagerAddOwnedPass(pm, mlirCreateTransformsCSE())
    success = mlirPassManagerRunOnOp(pm, MLIR.Operation.from_module(module))

    assert success
           |> beaverLogicalResultIsSuccess()
           |> Beaver.Native.to_term()

    mlirPassManagerDestroy(pm)
    mlirModuleDestroy(module)
    mlirContextDestroy(ctx)
  end

  test "Run a generic pass" do
    ctx = MLIR.Context.create()
    {:ok, module} = create_adder_module(ctx)
    type_id_allocator = mlirTypeIDAllocatorCreate()
    external = %MLIR.Pass{} = Beaver.Composer.create_pass(TestPass)
    pm = mlirPassManagerCreate(ctx)
    mlirPassManagerAddOwnedPass(pm, external)
    mlirPassManagerAddOwnedPass(pm, mlirCreateTransformsCSE())
    :ok = MLIR.PassManager.run(pm, module)
    :ok = MLIR.PassManager.destroy(pm)
    mlirModuleDestroy(module)
    mlirTypeIDAllocatorDestroy(type_id_allocator)
    mlirContextDestroy(ctx)
  end

  test "Run a func operation pass", %{ctx: ctx} do
    {:ok, module} = create_adder_module(ctx)
    external = %MLIR.Pass{} = Beaver.Composer.create_pass(TestFuncPass)
    pm = mlirPassManagerCreate(ctx)
    npm = mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create("func.func"))
    mlirOpPassManagerAddOwnedPass(npm, external)
    :ok = MLIR.PassManager.run(pm, module)
    :ok = MLIR.PassManager.destroy(pm)
    mlirModuleDestroy(module)
  end

  test "Run pass with patterns", %{ctx: ctx} do
    {:ok, module} = create_redundant_transpose_module(ctx)
    external = %MLIR.Pass{} = Beaver.Composer.create_pass(TestFuncPass)
    pm = mlirPassManagerCreate(ctx)
    npm = mlirPassManagerGetNestedUnder(pm, MLIR.StringRef.create("func.func"))
    mlirOpPassManagerAddOwnedPass(npm, external)
    :ok = MLIR.PassManager.run(pm, module)
    :ok = MLIR.PassManager.destroy(pm)
    mlirModuleDestroy(module)
  end

  defmacro some_constrain(_t) do
    true
  end

  defmacro has_one_use(_v) do
    true
  end

  def lower_to_llvm(ctx, module) do
    pm = mlirPassManagerCreate(ctx)
    mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass())
    mlirPassManagerAddOwnedPass(pm, mlirCreateConversionArithToLLVMConversionPass())
    status = mlirPassManagerRunOnOp(pm, module)

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
    {:ok, module} = MLIR.Module.create(module_str, ctx: ctx)
    MLIR.verify!(module)
    lower_to_llvm(ctx, MLIR.Operation.from_module(module))
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
        assert return |> Beaver.Native.to_term() == i * 2
      end)
    end
    |> Task.await_many()

    MLIR.ExecutionEngine.destroy(jit)
    MLIR.Module.destroy(module)
    mlirContextDestroy(ctx)
  end

  test "affine expr and map", %{ctx: ctx} do
    affine_dim_expr = mlirAffineDimExprGet(ctx, 0)
    affine_symbol_expr = mlirAffineSymbolExprGet(ctx, 1)

    exprs =
      Beaver.Native.array([affine_dim_expr, affine_symbol_expr], MLIR.AffineExpr, mut: true)

    map = mlirAffineMapGet(ctx, 3, 3, 2, exprs)
    txt = "(d0, d1, d2)[s0, s1, s2] -> (d0, s1)"
    assert map |> MLIR.to_string() == txt
    assert MLIR.Attribute.affine_map(map) |> MLIR.to_string() == "affine_map<#{txt}>"

    assert MLIR.AffineMap.create(3, 3, [MLIR.AffineMap.dim(0), MLIR.AffineMap.symbol(1)]).(ctx)
           |> MLIR.to_string() == txt
  end

  test "exception" do
    assert_raise Kinda.CallError, fn -> beaver_raw_to_string_Type(1) end
  end
end
