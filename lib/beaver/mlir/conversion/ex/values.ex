defmodule Beaver.MLIR.Conversion.Ex.Values do
  @moduledoc "SSA value lowering for Batata's owned handles; statuses and cleanup remain explicit."
  alias Beaver.{Changeset, MLIR, Walker}

  alias Beaver.MLIR.{
    Attribute,
    CAPI,
    ConversionPattern,
    ConversionPatternRewriter,
    Operation,
    RewriterBase,
    Type,
    Value
  }

  alias Beaver.MLIR.Conversion.{Ex.CRuntime, Plan}

  @structural ~w(func.func func.call func.return llvm.func cf.br cf.cond_br cf.switch scf.if scf.for scf.while scf.condition scf.yield scf.execute_region)

  def configure(plan, nil, _selected), do: plan

  def configure(plan, namespace, selected) do
    CRuntime.symbol("query", namespace)

    plan =
      Enum.reduce(
        @structural,
        plan,
        fn root, plan -> Plan.add_dynamically_legal_op(plan, root, &legal?/1, version: "1") end
      )

    roots =
      Enum.filter(selected, &String.starts_with?(&1, "ex.value_")) ++
        Enum.filter(~w(ex.func ex.call), &MapSet.member?(selected, &1)) ++ @structural

    Plan.add_pattern_population(
      plan,
      fn patterns, converter ->
        ctx = MLIR.context(patterns)

        for {from, to} <- CRuntime.value_types(),
            do:
              MLIR.TypeConverter.add_conversion_map(converter, [Type.get(from, ctx: ctx)], Type.get(to, ctx: ctx),
                fallback: :declined
              )

        for root <- Enum.sort(roots),
            do:
              ConversionPattern.add(patterns, root, converter, &lower(&1, &2, &3, converter),
                ctx: ctx,
                benefit: 2
              )

        :ok
      end,
      version: "values-v1/#{namespace}"
    )
  end

  defp legal?(op) do
    types =
      if Operation.name(op) == "func.func",
        do: [Attribute.value(op["function_type"])],
        else: Enum.map(Enum.to_list(Walker.operands(op)) ++ Enum.to_list(Walker.results(op)), &Value.type/1)

    block_types =
      for region <- Walker.regions(op),
          block <- Walker.blocks(region),
          arg <- Walker.arguments(block),
          do: Value.type(arg)

    if Enum.any?(types ++ block_types, fn t ->
         Enum.any?(Map.keys(CRuntime.value_types()), &String.contains?(MLIR.to_string(t), &1))
       end), do: :illegal, else: :legal
  end

  defp type(type, ctx) when is_binary(type), do: Type.get(CRuntime.value_types()[type] || type, ctx: ctx)
  defp type(type, ctx), do: type(MLIR.to_string(type), ctx)

  defp type_conversion(nil, _ctx), do: &Beaver.MLIR.Conversion.Ex.convert_type/1
  defp type_conversion(_converter, ctx), do: &type(&1, ctx)

  @doc false
  def lower(op, args, rewriter, converter \\ nil, namespace \\ nil) do
    base = ConversionPatternRewriter.as_base(rewriter)
    RewriterBase.set_insertion_point_before(base, op)

    emit = fn name, args, types ->
      %Changeset{name: name, context: MLIR.context(op), location: Operation.location(op)}
      |> Changeset.add_argument(args)
      |> Changeset.add_result(types)
      |> Operation.create()
      |> tap(&RewriterBase.insert(base, &1))
    end

    name = Operation.name(op)
    resolve = type_conversion(converter, MLIR.context(op))
    results = Enum.map(Walker.results(op), &resolve.(Value.type(&1)))

    replacement =
      case name do
        "ex.value_" <> call ->
          value_call(call, args, op, emit)

        name when name in ~w(ex.func func.func llvm.func) ->
          function(name, op, rewriter, converter, emit, resolve)

        name when name in ~w(ex.call func.call) ->
          call(op, args, results, emit)

        name when name in ~w(ex.return func.return) ->
          emit.("func.return", args, [])
          []

        "ex.btt_" <> call ->
          runtime_call(call, args, op, base, emit, namespace)

        "ex.var" ->
          raise ArgumentError, "ex.var without ex.bind is unsupported: #{MLIR.to_string(op)}"

        _ ->
          structural(op, args, results, rewriter, converter, emit)
      end

    unless replacement == :modified, do: ConversionPatternRewriter.replace_op(rewriter, op, replacement)
    :ok
  end

  defp runtime_call(name, args, op, base, emit, namespace) do
    symbol = CRuntime.symbol(name, namespace)
    root = Stream.iterate(op, &Operation.parent/1) |> Enum.find(&(Operation.parent(&1) |> MLIR.null?()))
    body = root |> Walker.regions() |> Enum.at(0) |> Walker.blocks() |> Enum.at(0)
    signature = CRuntime.function_type(name) |> Beaver.Deferred.resolve(MLIR.context(op))
    declaration = Enum.find(Walker.operations(body), &(&1["sym_name"] && Attribute.value(&1["sym_name"]) == symbol))

    if declaration do
      unless Operation.name(declaration) == "func.func" and
               MLIR.equal?(Attribute.value(declaration["function_type"]), signature),
             do: raise(ArgumentError, "incompatible Batata C ABI declaration: #{symbol}")
    else
      RewriterBase.set_insertion_point_to_end(base, body)

      emit.(
        "func.func",
        [
          [sym_name: Attribute.string(symbol), sym_visibility: Attribute.string("private"), function_type: signature],
          CAPI.mlirRegionCreate()
        ],
        []
      )
    end

    RewriterBase.set_insertion_point_before(base, op)

    emit.("func.call", [args, callee: Attribute.flat_symbol_ref(symbol)], Type.i32())
    |> Operation.results()
    |> Enum.to_list()
  end

  defp structural(op, args, results, rewriter, converter, emit) do
    regions = Enum.to_list(Walker.regions(op))
    new_regions = Enum.map(regions, fn _ -> CAPI.mlirRegionCreate() end)
    target = emit.(Operation.name(op), [args, Enum.to_list(Walker.successors(op)), new_regions], results)
    copy_attributes(op, target, [])

    for {old, new} <- Enum.zip(regions, Enum.to_list(Walker.regions(target))) do
      move_region(ConversionPatternRewriter.as_base(rewriter), old, new)
      convert_region!(rewriter, new, converter)
    end

    Enum.to_list(Operation.results(target))
  end

  defp call(op, args, results, emit) do
    if Operation.name(op) == "ex.call" and Attribute.value(op["arity"]) != length(args),
      do: raise(ArgumentError, "ex.call arity mismatch")

    call = emit.("func.call", [args, callee: Attribute.flat_symbol_ref(Attribute.value(op["callee"]))], results)
    copy_attributes(op, call, ~w(callee arity operandSegmentSizes))
    Enum.to_list(Operation.results(call))
  end

  defp value_call("load", args, op, emit) do
    result = op |> Operation.result(0) |> Value.type() |> type(MLIR.context(op))
    emit.("llvm.load", args, result) |> Operation.results() |> Enum.to_list()
  end

  defp value_call("store", args, _op, emit),
    do: emit.("llvm.store", args, []) |> Operation.results() |> Enum.to_list()

  defp value_call(name, args, op, emit) do
    # Bound temporary storage even when the call is nested in a loop.
    ptr = Type.llvm_pointer()
    saved = emit.("llvm.intr.stacksave", [], ptr) |> Operation.result(0)
    one = emit.("llvm.mlir.constant", [value: Attribute.integer(Type.i64(), 1)], Type.i64()) |> Operation.result(0)

    {[], operands, outputs} =
      Enum.reduce(CRuntime.value_parameters(name), {args, [], []}, fn
        {:in, _}, {[arg | rest], operands, outputs} ->
          {rest, operands ++ [arg], outputs}

        {direction, spelling}, {args, operands, outputs} ->
          type = type(spelling, MLIR.context(op))

          {value, args} =
            if direction == :out,
              do: {emit.("llvm.mlir.zero", [], type) |> Operation.result(0), args},
              else: {hd(args), tl(args)}

          slot = emit.("llvm.alloca", [one, elem_type: Attribute.type(type)], ptr) |> Operation.result(0)
          emit.("llvm.store", [value, slot], [])
          {args, operands ++ [slot], outputs ++ if(direction == :out, do: [{slot, type}], else: [])}
      end)

    status = emit.("ex.btt_" <> name, operands, Type.i32()) |> Operation.result(0)
    values = for {slot, type} <- outputs, do: emit.("llvm.load", [slot], type) |> Operation.result(0)
    emit.("llvm.intr.stackrestore", [saved], [])
    [status | values]
  end

  defp function("llvm.func", op, rewriter, converter, _emit, _resolve) do
    for region <- Walker.regions(op), do: convert_region!(rewriter, region, converter)
    :modified
  end

  defp function(_name, op, rewriter, converter, emit, resolve) do
    [region] = Enum.to_list(Walker.regions(op))
    ctx = MLIR.context(op)

    {target, target_region} =
      if Operation.name(op) == "ex.func" do
        [block] = Enum.to_list(Walker.blocks(region))
        terminator = CAPI.mlirBlockGetTerminator(block)
        if MLIR.null?(terminator), do: raise(ArgumentError, "ex.func body must end with a terminator (ex.return)")
        inputs = Enum.map(Walker.arguments(block), &resolve.(Value.type(&1)))
        outputs = Enum.map(Walker.operands(terminator), &resolve.(Value.type(&1)))

        target =
          emit.(
            "func.func",
            [[sym_name: op["sym_name"], function_type: Type.function(inputs, outputs)], CAPI.mlirRegionCreate()],
            []
          )

        [target_region] = Enum.to_list(Walker.regions(target))
        move_region(ConversionPatternRewriter.as_base(rewriter), region, target_region)
        copy_attributes(op, target, ~w(function_type))
        {target, target_region}
      else
        base = ConversionPatternRewriter.as_base(rewriter)
        spelling = MLIR.to_string(Attribute.value(op["function_type"]))
        spelling = Enum.reduce(CRuntime.value_types(), spelling, fn {from, to}, s -> String.replace(s, from, to) end)
        RewriterBase.start_op_modification(base, op)
        Operation.put_attribute(op, "function_type", Attribute.type(Type.get(spelling, ctx: ctx)))
        RewriterBase.finalize_op_modification(base, op)
        {op, region}
      end

    convert_region!(rewriter, target_region, converter)

    if target == op, do: :modified, else: []
  end

  defp move_region(base, from, to) do
    # Rewriter moves let failed conversions restore the original region body.
    anchor = MLIR.Block.create()
    CAPI.mlirRegionAppendOwnedBlock(to, anchor)
    RewriterBase.inline_region_before(base, from, anchor)
    RewriterBase.erase_block(base, anchor)
  end

  defp convert_region!(_rewriter, _region, nil), do: :ok

  defp convert_region!(rewriter, region, converter) do
    unless ConversionPatternRewriter.convert_region_types(rewriter, region, converter) |> MLIR.LogicalResult.success?(),
      do: raise(ArgumentError, "Batata value region signature conversion failed")
  end

  defp copy_attributes(from, to, except) do
    for {id, value} <- Walker.attributes(from),
        name = id |> CAPI.mlirIdentifierStr() |> MLIR.to_string(),
        name not in except,
        do: Operation.put_attribute(to, name, value)
  end
end
