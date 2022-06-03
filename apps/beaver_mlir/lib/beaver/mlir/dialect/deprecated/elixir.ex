defmodule Beaver.MLIR.Dialect.Elixir.Deprecated do
  @moduledoc false
  require Logger
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI.IR

  defmacro __using__(_opts) do
    quote do
      @before_compile unquote(__MODULE__)
      Module.register_attribute(__MODULE__, :mlir, accumulate: false, persist: true)
      Module.register_attribute(__MODULE__, :mlir_spec, accumulate: true, persist: true)
      Module.register_attribute(__MODULE__, :before_compile, accumulate: false, persist: true)
    end
  end

  defmodule Function.Block.Context do
    # ret is the return value of the block
    @enforce_keys [:vars, :ops, :ret]
    defstruct vars: %{}, ops: [], ret: nil
  end

  def create_op(
        ast = {:=, [line: _], [{var_name, [version: version, line: _], nil}, value]},
        %Function.Block.Context{vars: vars, ops: ops},
        {_args, func_body, ctx}
      )
      when is_integer(value) do
    const_op =
      ctx
      |> MLIR.Operation.State.get!("arith.constant")
      |> MLIR.Operation.State.add_attr(value: "#{value} : i64")
      |> MLIR.Operation.State.add_result(["i64"])
      |> MLIR.Operation.create()

    IR.mlirBlockInsertOwnedOperation(func_body, 0, const_op)
    res = IR.mlirOperationGetResult(const_op, 0)

    {ast,
     %Function.Block.Context{
       vars: Map.put(vars, {var_name, version}, res),
       ops: ops ++ [const_op],
       ret: {const_op, [res]}
     }}
  end

  def create_op(
        ast =
          {:=, [line: _],
           [
             {var_name, [version: var_version, line: _], nil},
             {func_symbol, [line: _],
              [
                {arg_0, [version: arg_0_version, line: _], nil},
                {arg_1, [version: arg_1_version, line: _], nil}
              ]}
           ]},
        %Function.Block.Context{vars: vars, ops: ops},
        {_args, func_body, ctx}
      ) do
    arg1 = vars |> Map.get({arg_0, arg_0_version})
    arg2 = vars |> Map.get({arg_1, arg_1_version})

    results = ["i64"]

    call_op =
      ctx
      |> MLIR.Operation.State.get!("func.call")
      |> MLIR.Operation.State.add_attr(callee: "@#{func_symbol}")
      |> MLIR.Operation.State.add_operand([arg1, arg2])
      |> MLIR.Operation.State.add_result(results)
      |> MLIR.Operation.create()

    IR.mlirBlockInsertOwnedOperation(func_body, length(ops), call_op)
    res = IR.mlirOperationGetResult(call_op, 0)

    {ast,
     %Function.Block.Context{
       vars: Map.put(vars, {var_name, var_version}, res),
       ops: ops ++ [call_op],
       ret: {call_op, [res]}
     }}
  end

  def create_op(
        ast =
          {{:., [line: _], [IO, :puts]}, [line: _],
           [{arg_0, [version: arg_0_version, line: _], nil}]},
        %Function.Block.Context{vars: vars, ops: ops},
        {_args, func_body, ctx}
      ) do
    arg1 = vars |> Map.get({arg_0, arg_0_version})

    call_op =
      ctx
      |> MLIR.Operation.State.get!("func.call")
      |> MLIR.Operation.State.add_attr(callee: "@printI64")
      |> MLIR.Operation.State.add_operand([arg1])
      |> MLIR.Operation.State.add_result([])
      |> MLIR.Operation.create()

    IR.mlirBlockInsertOwnedOperation(func_body, length(ops), call_op)

    {ast,
     %Function.Block.Context{
       vars: vars,
       ops: ops ++ [call_op],
       ret: {call_op, []}
     }}
  end

  def create_op(
        ast =
          {{:., [line: _], [:erlang, :+]}, [line: _],
           [{lhs, [version: lhs_v, line: _], nil}, {rhs, [version: rhs_v, line: _], nil}]},
        %Function.Block.Context{vars: vars, ops: ops},
        {args, func_body, ctx}
      ) do
    lhs = args |> Map.get({lhs, lhs_v})
    rhs = args |> Map.get({rhs, rhs_v})

    results = ["i64"]

    add_op =
      ctx
      |> MLIR.Operation.State.get!("elixir.add")
      |> MLIR.Operation.State.add_operand([lhs, rhs])
      |> MLIR.Operation.State.add_result(results)
      |> MLIR.Operation.create()

    IR.mlirBlockInsertOwnedOperation(func_body, 0, add_op)

    results =
      for n <- 0..(length(results) - 1)//1 do
        IR.mlirOperationGetResult(add_op, n)
      end

    {ast,
     %Function.Block.Context{
       vars: vars,
       ops: ops ++ [add_op],
       ret: {add_op, results}
     }}
  end

  def create_op(ast, acc, _) do
    # Logger.warn("ignore: #{inspect(ast, pretty: true)}")
    {ast, acc}
  end

  def create_func(
        {
          name,
          spec,
          file,
          {:v1, :def, __meta, [{[line: line], arguments, _guards = [], clause}]}
        },
        ctx,
        module
      ) do
    location = MLIR.Location.get!(ctx, file, line)

    {sym_visibility, function_type, funcBodyArgTypes, funcBodyArgLocs} =
      if name == :main do
        {"public", spec, [], []}
      else
        {"private", spec, [], []}
      end

    funcBodyRegion = IR.mlirRegionCreate()
    funcBody = MLIR.Block.create(funcBodyArgTypes, funcBodyArgLocs)

    %MLIR.Dialect.Builtin.Deprecated.Type.Function{inputs: inputs} =
      MLIR.Dialect.Builtin.Deprecated.Type.Function.from_attr(ctx, function_type)

    args =
      for {{name, [version: verison, line: line], nil}, input} <- Enum.zip(arguments, inputs) do
        # TODO: use real type
        arg_loc = MLIR.Location.get!(ctx, file, line)

        {{name, verison},
         funcBody |> MLIR.Block.add_arg!(ctx, [{input, arg_loc}]) |> List.first()}
      end
      |> Map.new()

    IR.mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody)
    # create func
    operation_state = MLIR.Operation.State.get!("func.func", location)

    MLIR.Operation.State.add_attr(operation_state,
      sym_name: "\"#{name}\"",
      sym_visibility: "\"#{sym_visibility}\"",
      function_type: function_type
    )

    MLIR.Operation.State.add_regions(operation_state, [funcBodyRegion])
    func_op = operation_state |> MLIR.Operation.create()
    MLIR.Module.verify!(module)
    moduleBody = IR.mlirModuleGetBody(module)

    context = struct(Function.Block.Context)

    {_, context} =
      clause
      |> Macro.postwalk(context, fn ast, acc -> create_op(ast, acc, {args, funcBody, ctx}) end)

    %Function.Block.Context{ret: ret, ops: ops} = context
    # TODO: fix this workaround

    {_op, returns} =
      case {ret, name} do
        {_, :main} ->
          {nil, []}

        {nil, _} ->
          {nil, []}

        {{op, returns}, _} ->
          {op, returns}
      end

    return_op =
      ctx
      |> MLIR.Operation.State.get!("func.return")
      |> MLIR.Operation.State.add_operand(returns)
      |> MLIR.Operation.create()

    IR.mlirBlockInsertOwnedOperation(funcBody, length(ops), return_op)

    IR.mlirBlockInsertOwnedOperation(moduleBody, 0, func_op)
  end

  defmacro __before_compile__(env) do
    mlir = Module.get_attribute(env.module, :mlir)
    mlir_specs = Module.get_attribute(env.module, :mlir_spec) |> Enum.reverse()

    # TODO: support adding more than one type spec for function
    if length(mlir) != length(mlir_specs) do
      raise "length of @mlir exports and number of @mlir_spec must be equal"
    end

    # create module
    MLIR.CAPI.call_to_load_code()
    MLIR.CAPI.IR.load(Beaver.MLIR.CAPI)
    MLIR.CAPI.Registration.load(Beaver.MLIR.CAPI)
    ctx = IR.mlirContextCreate()
    MLIR.CAPI.Registration.register_elixir_dialect(ctx)
    MLIR.CAPI.Registration.mlirRegisterAllDialects(ctx)

    with_helper = """
    module {
      func.func private @printNewline()
      func.func private @printI64(i64)
    }
    """

    module = MLIR.Module.create!(ctx, with_helper)

    for {fun = {name, _arity}, spec} <- Enum.zip(mlir, mlir_specs) do
      create_func({name, spec, env.file, Module.get_definition(env.module, fun)}, ctx, module)
    end

    is_verified = IR.mlirOperationVerify(module) |> Exotic.Value.extract()

    if not is_verified do
      raise "fail to verify mlir module"
    end

    MLIR.CAPI.IR.mlirContextDestroy(ctx)
    []
  end
end
