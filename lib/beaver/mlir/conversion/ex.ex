alias Beaver.MLIR.{Attribute, CAPI, ConversionPatternRewriter, Operation, RewriterBase, Type, Value}

alias Beaver.Native

defmodule Beaver.MLIR.Conversion.Ex do
  @moduledoc """
  Reference lowering for the language, execution and runtime layers of `ex`.

  `plan/1` lowers all layers to `func`/`arith`/`scf`/`cf`. With `:layers`, it
  converts only the selected owners and retains the other Ex operations.
  Intermediate plans leave explicit type bridges; finish with `finish!/1`
  before passing the result to LLVM or another target conversion.

  `RuntimeABI` owns the original Batata tagged-word symbols and signatures.
  This reference path does not target wavefrontier/batata's current C runtime.
  Materialize `ex.var`/`ex.bind` and expand supported `ex.case` operations before
  conversion. See the [Ex organization guide](ex-dialect.html).
  """

  alias Beaver.Changeset
  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Ex.RuntimeABI
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Dialect.Ex, as: ExDialect
  alias Beaver.Walker

  @term_types ~w(!ex.term !ex.bound !ex.unbound)

  @doc "Completes remaining Ex lowering and reconciles the type bridges left by layer plans."
  def finish!(ir) do
    ir =
      ir
      |> then(&Plan.run!(plan(), &1))
      |> Beaver.Composer.append("reconcile-unrealized-casts")
      |> Beaver.Composer.run!()

    Walker.prewalk(ir, fn
      %Operation{} = op ->
        if Operation.name(op) == "builtin.unrealized_conversion_cast",
          do: raise(ArgumentError, "unresolved Ex type bridge")

        op

      element ->
        element
    end)

    ir
  end

  @doc "Returns a reference conversion plan for all Ex owners, or only the selected `:layers`."
  @spec plan(keyword()) :: Plan.t()
  def plan(opts \\ []) do
    {layers, opts} = Keyword.pop(opts, :layers, ExDialect.layers())
    layers = List.wrap(layers)

    unless layers != [] and layers -- ExDialect.layers() == [],
      do: raise(ArgumentError, "invalid Ex conversion layers: #{inspect(layers)}")

    selected = Enum.flat_map(layers, &ExDialect.operations/1) |> MapSet.new()

    defaults = [mode: :full, folding_mode: :after_patterns, build_materializations: length(layers) == 3]

    Plan.new(Keyword.merge(defaults, opts))
    |> Plan.add_legal_dialect("builtin")
    |> Plan.add_legal_dialect("func")
    |> Plan.add_legal_dialect("arith")
    |> Plan.add_legal_dialect("cf")
    |> Plan.add_legal_dialect("scf")
    |> Plan.add_legal_dialect("llvm")
    |> Plan.add_illegal_dialect("ex")
    |> retain_other_layers(selected)
    |> Plan.add_conversion_map(@term_types, "i64")
    |> Plan.add_pattern_population(&populate_native_scalar_patterns/2, version: "1.0")
    |> add_conversion_patterns(selected)
  end

  defp retain_other_layers(plan, selected),
    do: Enum.reduce(ExDialect.operations() -- MapSet.to_list(selected), plan, &Plan.add_legal_op(&2, &1))

  defp add_conversion_patterns(plan, selected) do
    patterns = [
      {~w(call), &convert_call/3},
      {~w(return), &convert_return/3},
      {~w(var), &convert_var/3},
      {~w(func), &convert_func/3},
      {~w(try), &convert_try/3},
      {~w(make_fun), &convert_make_fun/3},
      {~w(make_fun_with_arity), &convert_make_fun_with_arity/3},
      {~w(make_fun_with_signature), &convert_make_fun_with_signature/3},
      {~w(apply), &convert_apply/3},
      {~w(tuple list map binary), &convert_collection/3},
      {~w(is_integer is_float is_atom is_binary is_list is_tuple is_map), &convert_term_predicate/3},
      {~w(func_addr), &convert_func_addr/3}
    ]

    specialized = Map.new(for {names, callback} <- patterns, name <- names, do: {"ex." <> name, callback})
    intrinsics = MapSet.new(RuntimeABI.intrinsics(), fn {name, _} -> "ex.#{name}" end)

    Enum.reduce(Enum.sort(selected), plan, fn root, plan ->
      callback = specialized[root] || if(MapSet.member?(intrinsics, root), do: &convert_term_read/3)
      if callback, do: Plan.add_conversion_pattern(plan, root, callback, version: "1.0"), else: plan
    end)
  end

  defp populate_native_scalar_patterns(patterns, converter),
    do: CAPI.beaverPopulateExScalarConversionPatterns(patterns.ref, converter.ref)

  @term_intrinsics RuntimeABI.intrinsics()

  @doc "Returns the reference runtime symbols; see `RuntimeABI`."
  @spec term_intrinsic_symbols() :: [String.t()]
  defdelegate term_intrinsic_symbols(), to: RuntimeABI, as: :symbols

  @doc """
  Converts an `ex` term type to its scalar word representation.
  """
  @spec convert_type(Type.t()) :: Type.t()
  def convert_type(type), do: if(term_type?(type), do: Type.integer(64, ctx: MLIR.context(type)), else: type)

  defp convert_collection(op, operands, rewriter) do
    base = insertion_point(op, rewriter)
    name = Operation.name(op)

    if name == "ex.map" and rem(length(operands), 2) != 0,
      do: raise(ArgumentError, "ex.map requires an even number of key/value operands, got #{length(operands)}")

    list = build_list(operands, op, rewriter, base)

    symbol =
      %{
        "ex.tuple" => @term_intrinsics.tuple_from_list,
        "ex.map" => @term_intrinsics.map_from_list,
        "ex.binary" => @term_intrinsics.binary_from_list
      }[name]

    result = if symbol, do: emit_runtime_call(op, rewriter, base, symbol, [list]), else: list
    replace_with(rewriter, op, result)
  end

  defp convert_term_predicate(op, [operand], rewriter) do
    base = insertion_point(op, rewriter)
    [original] = Enum.to_list(Walker.operands(op))
    word = maybe_tag(original, operand, op, base)
    intrinsic = op |> Operation.name() |> read_intrinsic()

    result = emit_runtime_call(op, rewriter, base, intrinsic, [word])
    replace_with(rewriter, op, result)
  end

  # `ex.func_addr` resolves a function symbol to its address (an i64 word):
  # func.constant yields the function value, which func-to-llvm lowers to a
  # pointer. Used to hand a compiled reducer to the runtime's
  # arbitrary-closure enumerable reduce.
  defp convert_func_addr(op, [], rewriter) do
    context = MLIR.context(op)
    base = insertion_point(op, rewriter)
    name = op |> required_attribute("sym_name") |> CAPI.mlirStringAttrGetValue() |> MLIR.to_string()
    result_type = op |> Operation.result(0) |> Value.type()

    func_const =
      create_operation(op, "func.constant", [value: Attribute.flat_symbol_ref(name, ctx: context)], result_type)

    RewriterBase.insert(base, func_const)
    replace_with(rewriter, op, Operation.result(func_const, 0))
  end

  # `ex.try` lowers to a setjmp/longjmp pair: allocate a jmp_buf on the stack,
  # push it on the runtime's buffer stack, and call libc `setjmp`. The normal
  # path runs the body region; a `throw` longjmps back and the catch region
  # matches the thrown value.
  defp convert_try(op, [], rewriter) do
    context = MLIR.context(op)
    base = insertion_point(op, rewriter)
    [body_region, catch_region] = op |> Walker.regions() |> Enum.to_list()

    size = emit_runtime_call(op, rewriter, base, @term_intrinsics.jmp_buf_size, [])

    alloca =
      create_operation(
        op,
        "llvm.alloca",
        [
          [size],
          [elem_type: Attribute.type(Type.integer(8, ctx: context)), alignment: Attribute.integer(Type.i64(), 8)]
        ],
        Type.llvm_pointer(ctx: context)
      )

    RewriterBase.insert(base, alloca)
    buf = Operation.result(alloca, 0)

    push = emit_runtime_call(op, rewriter, base, @term_intrinsics.try_push, [buf])
    _ = push

    # libc's setjmp is resolved indirectly (the ORC linker does not resolve
    # libc symbols on its own): the runtime exposes its address.
    setjmp_addr = emit_runtime_call(op, rewriter, base, @term_intrinsics.setjmp_addr, [])

    fn_ptr = create_operation(op, "llvm.inttoptr", [setjmp_addr], Type.llvm_pointer(ctx: context))

    RewriterBase.insert(base, fn_ptr)

    setjmp =
      create_operation(
        op,
        "llvm.call",
        [
          [Operation.result(fn_ptr, 0), buf],
          [
            operandSegmentSizes: Attribute.dense_array([2, 0], Native.I32, ctx: context),
            op_bundle_sizes: Attribute.dense_array([], Native.I32, ctx: context)
          ]
        ],
        Type.i64()
      )

    RewriterBase.insert(base, setjmp)
    saved = Operation.result(setjmp, 0)

    zero = emit_constant(0, op, base) |> Operation.result(0)

    cmp = create_operation(op, "arith.cmpi", [[saved, zero], [predicate: cmp_i_predicate_attr(0)]], Type.i1())

    RewriterBase.insert(base, cmp)

    scf_if =
      create_operation(
        op,
        "scf.if",
        [Operation.result(cmp, 0), CAPI.mlirRegionCreate(), CAPI.mlirRegionCreate()],
        [Type.i64()]
      )

    [new_then, new_else] = scf_if |> Walker.regions() |> Enum.to_list()
    CAPI.mlirRegionTakeBody(new_then, body_region)
    CAPI.mlirRegionTakeBody(new_else, catch_region)

    insert_try_pop(op, rewriter, new_else, :start)
    insert_try_pop(op, rewriter, new_then, :end)

    RewriterBase.set_insertion_point_before(base, op)
    RewriterBase.insert(base, scf_if)

    ConversionPatternRewriter.replace_op(rewriter, op, scf_if |> Walker.results() |> Enum.to_list())

    :ok
  end

  # Catch pops before matching; the normal body pops immediately before its terminator.
  defp insert_try_pop(op, rewriter, region, position) do
    ensure_intrinsic_declaration(op, rewriter, @term_intrinsics.try_pop)
    [block] = Enum.to_list(Walker.blocks(region))
    base = ConversionPatternRewriter.as_base(rewriter)

    case position do
      :start -> RewriterBase.set_insertion_point_to_start(base, block)
      :end -> RewriterBase.set_insertion_point_before(base, CAPI.mlirBlockGetTerminator(block))
    end

    RewriterBase.insert(base, runtime_call(op, @term_intrinsics.try_pop, []))
    :ok
  end

  # All closure ABIs share environment padding, validation, and emission.
  # Keep the legacy constructor's six words and the newer metadata in ABI order.
  defp convert_make_fun(op, operands, rewriter),
    do: convert_closure(op, operands, rewriter, ~w(fn_idx env_len))

  defp convert_make_fun_with_arity(op, operands, rewriter),
    do: convert_closure(op, operands, rewriter, ~w(fn_idx arity env_len))

  defp convert_make_fun_with_signature(op, operands, rewriter),
    do: convert_closure(op, operands, rewriter, ~w(fn_idx arity result_mode env_len))

  defp convert_closure(op, operands, rewriter, attributes) do
    base = insertion_point(op, rewriter)
    name = Operation.name(op)

    values =
      Enum.map(attributes, fn attribute ->
        op |> required_attribute(attribute) |> CAPI.mlirIntegerAttrGetValueInt() |> Native.to_term()
      end)

    metadata = Map.new(Enum.zip(attributes, values))
    env_len = metadata["env_len"]

    unless length(operands) == env_len,
      do: raise(ArgumentError, "#{name} env_len #{env_len} does not match #{length(operands)} operands")

    if Map.has_key?(metadata, "arity") and metadata["arity"] not in 0..4,
      do: raise(ArgumentError, "#{name} supports arities 0..4, got #{metadata["arity"]}")

    if Map.has_key?(metadata, "result_mode") and metadata["result_mode"] not in 0..1,
      do: raise(ArgumentError, "#{name} supports result modes 0..1, got #{metadata["result_mode"]}")

    constants = Enum.map(values, &(emit_constant(&1, op, base) |> Operation.result(0)))
    pad = List.duplicate(emit_constant(0, op, base) |> Operation.result(0), 4 - env_len)

    closure = emit_runtime_call(op, rewriter, base, read_intrinsic(name), constants ++ operands ++ pad)

    replace_with(rewriter, op, closure)
  end

  # Applies a first-class function value: reads the function index and env
  # words from the closure, then dispatches to the matching `__fn_*`.
  defp convert_apply(op, operands, rewriter) do
    base = insertion_point(op, rewriter)
    [closure | args] = operands

    arg_count = op |> required_attribute("arg_count") |> CAPI.mlirIntegerAttrGetValueInt() |> Native.to_term()

    unless length(args) == arg_count,
      do: raise(ArgumentError, "ex.apply arg_count #{arg_count} does not match #{length(args)} arguments")

    idx = emit_runtime_call(op, rewriter, base, @term_intrinsics.fun_idx, [closure])

    envs =
      for i <- 0..3 do
        index_const = emit_constant(i, op, base) |> Operation.result(0)

        emit_runtime_call(op, rewriter, base, @term_intrinsics.fun_env, [closure, index_const])
      end

    pad = List.duplicate(emit_constant(0, op, base) |> Operation.result(0), 4 - arg_count)

    dispatch = runtime_call(op, "__fn_dispatch", [idx] ++ envs ++ args ++ pad)

    RewriterBase.insert(base, dispatch)
    RewriterBase.set_insertion_point_after(base, dispatch)
    replace_with(rewriter, op, Operation.result(dispatch, 0))
  end

  defp convert_term_read(op, operands, rewriter) do
    base = insertion_point(op, rewriter)
    symbol = read_intrinsic(Operation.name(op))
    result = emit_runtime_call(op, rewriter, base, symbol, operands)
    replace_with(rewriter, op, result)
  end

  for {name, symbol} <- @term_intrinsics do
    defp read_intrinsic(unquote("ex." <> Atom.to_string(name))), do: unquote(symbol)
  end

  defp insertion_point(op, rewriter) do
    base = ConversionPatternRewriter.as_base(rewriter)
    RewriterBase.set_insertion_point_before(base, op)
    base
  end

  defp replace_with(rewriter, op, value) do
    ConversionPatternRewriter.replace_op(rewriter, op, value)
    :ok
  end

  defp maybe_tag(original, converted, op, base),
    do: if(term_type?(Value.type(original)), do: converted, else: tag_integer(converted, op, base))

  # Tags a scalar i64 as an immediate integer term: word = value << 3 with the
  # low 3 bits (tag 0b000) left as the integer tag.
  defp tag_integer(value, op, base) do
    three = emit_constant(3, op, base)

    shl = create_operation(op, "arith.shli", [value, Operation.result(three, 0)], Type.i64())

    RewriterBase.insert(base, shl)
    RewriterBase.set_insertion_point_after(base, shl)
    Operation.result(shl, 0)
  end

  # Builds a proper list word by consing words onto nil. Nil is the atom term
  # with id 0: tag 0b001 | (0 << 3) = 1.
  defp build_list([], op, _rewriter, base),
    do: emit_constant(1, op, base) |> Operation.result(0)

  defp build_list(words, op, rewriter, base) do
    ensure_intrinsic_declaration(op, rewriter, @term_intrinsics.list_cons)
    RewriterBase.set_insertion_point_before(base, op)

    words
    |> Enum.reverse()
    |> Enum.reduce(emit_constant(1, op, base) |> Operation.result(0), fn word, tail ->
      emit_declared_runtime_call(op, base, @term_intrinsics.list_cons, [word, tail])
    end)
  end

  defp create_operation(anchor, name, arguments, results \\ []) do
    %Changeset{name: name, context: MLIR.context(anchor), location: Operation.location(anchor)}
    |> Changeset.add_argument(arguments)
    |> Changeset.add_result(results)
    |> Operation.create()
  end

  defp runtime_call(op, symbol, args, result_type \\ Type.i64()) do
    callee = Attribute.flat_symbol_ref(symbol, ctx: MLIR.context(op))
    create_operation(op, "func.call", [args, callee: callee], result_type)
  end

  defp emit_constant(value, op, base) do
    constant = create_operation(op, "arith.constant", [value: Attribute.integer(Type.i64(), value)], Type.i64())

    RewriterBase.insert(base, constant)
    RewriterBase.set_insertion_point_after(base, constant)
    constant
  end

  defp emit_runtime_call(op, rewriter, base, symbol, args) do
    ensure_intrinsic_declaration(op, rewriter, symbol)
    emit_declared_runtime_call(op, base, symbol, args)
  end

  defp emit_declared_runtime_call(op, base, symbol, args) do
    RewriterBase.set_insertion_point_before(base, op)

    call = runtime_call(op, symbol, args)

    RewriterBase.insert(base, call)
    RewriterBase.set_insertion_point_after(base, call)
    Operation.result(call, 0)
  end

  # `func.call` requires the callee symbol to exist, so each intrinsic gets a
  # public `func.func` declaration (no body) at module scope. At link time the
  # Zig term runtime shared library satisfies the symbols.
  defp ensure_intrinsic_declaration(operation, rewriter, symbol) do
    module_op =
      Stream.iterate(operation, &Operation.parent/1) |> Enum.find(fn op -> op |> Operation.parent() |> MLIR.null?() end)

    body = module_body(module_op)

    unless declaration_exists?(body, symbol) do
      base = ConversionPatternRewriter.as_base(rewriter)
      RewriterBase.set_insertion_point_to_end(base, body)

      declaration =
        create_operation(operation, "func.func", [
          [sym_name: Attribute.string(symbol)],
          [sym_visibility: Attribute.string("private")],
          [function_type: RuntimeABI.function_type(symbol, MLIR.context(operation))],
          CAPI.mlirRegionCreate()
        ])

      RewriterBase.insert(base, declaration)
    end

    :ok
  end

  defp declaration_exists?(body, symbol) do
    body
    |> Walker.operations()
    |> Enum.any?(fn op ->
      Operation.name(op) == "func.func" and
        case op |> Operation.fetch("sym_name") do
          {:ok, attribute} -> attribute |> CAPI.mlirStringAttrGetValue() |> MLIR.to_string() == symbol
          :error -> false
        end
    end)
  end

  defp module_body(module_op),
    do: module_op |> Walker.regions() |> Enum.to_list() |> hd() |> Walker.blocks() |> Enum.to_list() |> hd()

  defp term_type?(type), do: MLIR.to_string(type) in @term_types

  defp cmp_i_predicate_attr(i), do: Attribute.integer(Type.i64(), i)

  defp convert_call(op, args, rewriter) do
    base = insertion_point(op, rewriter)

    callee = op |> required_attribute("callee") |> CAPI.mlirStringAttrGetValue() |> MLIR.to_string()

    arity = op |> required_attribute("arity") |> CAPI.mlirIntegerAttrGetValueInt() |> Native.to_term()

    unless length(args) == arity,
      do: raise(ArgumentError, "ex.call arity attribute #{arity} does not match #{length(args)} arguments")

    result_type = op |> Operation.result(0) |> Value.type() |> convert_type()

    call = runtime_call(op, callee, args, result_type)

    RewriterBase.insert(base, call)
    replace_with(rewriter, op, Operation.result(call, 0))
  end

  defp convert_return(op, operands, rewriter) do
    base = insertion_point(op, rewriter)

    return_op = create_operation(op, "func.return", operands)

    RewriterBase.insert(base, return_op)
    replace_with(rewriter, op, return_op)
  end

  defp convert_var(op, [], _rewriter),
    do: raise(ArgumentError, "ex.var without ex.bind is unsupported: #{inspect(MLIR.to_string(op))}")

  defp convert_func(op, [], rewriter) do
    base = insertion_point(op, rewriter)

    sym_name = op |> required_attribute("sym_name") |> CAPI.mlirStringAttrGetValue() |> MLIR.to_string()

    [body_region] = op |> Walker.regions() |> Enum.to_list()
    [block] = body_region |> Walker.blocks() |> Enum.to_list()
    terminator = CAPI.mlirBlockGetTerminator(block)

    if MLIR.null?(terminator), do: raise(ArgumentError, "ex.func body must end with a terminator (ex.return)")

    return_types = terminator |> Walker.operands() |> Enum.map(&(&1 |> Value.type() |> convert_type()))

    arg_types = block |> Walker.arguments() |> Enum.to_list() |> Enum.map(&(&1 |> Value.type() |> convert_type()))

    function_type = Type.function(arg_types, return_types)

    func =
      create_operation(op, "func.func", [
        [sym_name: Attribute.string(sym_name)],
        [function_type: function_type],
        CAPI.mlirRegionCreate()
      ])

    func_body = func |> Walker.regions() |> Enum.to_list() |> hd()
    CAPI.mlirRegionTakeBody(func_body, body_region)

    RewriterBase.insert(base, func)
    replace_with(rewriter, op, func)
  end

  defp required_attribute(op, name) do
    case Operation.fetch(op, name) do
      {:ok, attribute} -> attribute
      :error -> raise ArgumentError, "#{Operation.name(op)} requires attribute #{inspect(name)}"
    end
  end
end
