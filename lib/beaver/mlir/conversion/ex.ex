alias Beaver.MLIR.{Attribute, CAPI, ConversionPatternRewriter, Operation, RewriterBase, Type, Value}

alias Beaver.Native

defmodule Beaver.MLIR.Conversion.Ex do
  @moduledoc """
  Conversion plan for lowering the `ex` dialect to `func`/`arith`/`scf`/`cf`.

  M1 lowering for the scalar subset of the `ex` dialect:

    * `ex.lit` -> `arith.constant`
    * `ex.add` -> `arith.addi`
    * `ex.sub` -> `arith.subi`
    * `ex.mul` -> `arith.muli`
    * `ex.cmp` -> `arith.cmpi`
    * `ex.if` -> `scf.if`
    * `ex.yield` -> `scf.yield`
    * `ex.call` -> `func.call`
    * `ex.return` -> `func.return`
    * `ex.func` -> `func.func` (body region moved, argument and `ex.return`
      types converted)

  Term-universe ops (`ex.tuple`/`ex.list`/`ex.map`/`ex.binary` and the
  `ex.is_*` predicates) convert to calls into the Zig term runtime ABI, which
  is implemented in the compiler repo (batata) rather than Beaver. The
  declaration-first manifest (symbol names and C signatures) is mirrored in
  `@term_intrinsics`; batata implements exactly these symbols.

  The `ex` term types (`!ex.dyn`/`!ex.bound`/`!ex.unbound`) convert to a
  64-bit tagged word (`i64`): low 3 bits hold the tag, the rest the payload.
  Scalar integers are tagged in-place with `arith.shli` when they cross into
  a term construction; values that are already term-typed pass through.
  Containers are built with a fixed-arity cons chain followed by a
  `*_from_list` intrinsic, so no variadic ABI or stack buffers are needed.

  `plan/1` returns a reusable `Beaver.MLIR.Conversion.Plan`; run it with
  `Beaver.MLIR.Conversion.Plan.run/2` and continue with the standard
  `arith-to-llvm` and `func-to-llvm` passes.

  `ex.var`/`ex.bind` must be materialized to SSA first with
  `Beaver.MLIR.Dialect.Ex.MaterializeBoundVariables`.
  """

  alias Beaver.Changeset
  alias Beaver.MLIR
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.Walker

  @term_types ~w(!ex.term !ex.bound !ex.unbound)

  @doc """
  Returns a conversion plan lowering the `ex` scalar subset to `func`/`arith`.
  """
  @spec plan(keyword()) :: Plan.t()
  def plan(opts \\ []) do
    Plan.new(Keyword.merge([mode: :full, folding_mode: :after_patterns, build_materializations: true], opts))
    |> Plan.add_legal_dialect("builtin")
    |> Plan.add_legal_dialect("func")
    |> Plan.add_legal_dialect("arith")
    |> Plan.add_legal_dialect("cf")
    |> Plan.add_legal_dialect("scf")
    |> Plan.add_legal_dialect("llvm")
    |> Plan.add_illegal_dialect("ex")
    |> Plan.add_conversion_map(@term_types, "i64")
    |> Plan.add_pattern_population(&populate_native_scalar_patterns/2, version: "1.0")
    |> add_conversion_patterns()
  end

  defp add_conversion_patterns(plan) do
    patterns = [
      {~w(call), &convert_call/3},
      {~w(return), &convert_return/3},
      {~w(var), &convert_var/3},
      {~w(func), &convert_func/3},
      {~w(self), &convert_self/3},
      {~w(send), &convert_send/3},
      {~w(receive), &convert_receive/3},
      {~w(mailbox_clear), &convert_mailbox_clear/3},
      {~w(
        spawn runtime_create runtime_enter runtime_leave runtime_destroy result_create result_create_term
        result_destroy result_root_kind result_root_word result_exception_kind result_exception_reason
        result_term_kind result_atom_name result_term_length result_term_get term_export term_import
        exported_clone exported_destroy exported_length exported_get term_handle_export term_handle_destroy
        process_table_reset cont_save receive_cont_save cont_pending cont_active cont_clear cont_load_arg
        cont_load_acc cont_load_cursor schedule_next worker_run process_wait mailbox_len mailbox_peek
        mailbox_remove nil_word monotonic_time receive_start receive_start_set native_time unique_integer
        current_entry process_done process_exit process_exit_reason process_trap_exit process_dictionary_get
        process_dictionary_put link unlink exit monitor demonitor processes_runnable process_result
      ), &convert_term_read/3},
      {~w(to_int), &convert_to_int/3},
      {~w(reduction_tick clock_init yield_mark), &convert_term_read/3},
      {~w(try), &convert_try/3},
      {~w(throw), &convert_throw/3},
      {~w(raise), &convert_raise/3},
      {~w(catch_value), &convert_catch_value/3},
      {~w(make_fun), &convert_make_fun/3},
      {~w(make_fun_with_arity), &convert_make_fun_with_arity/3},
      {~w(make_fun_with_signature), &convert_make_fun_with_signature/3},
      {~w(fun_arity fun_result_mode), &convert_term_read/3},
      {~w(apply), &convert_apply/3},
      {~w(tuple), &convert_term_tuple/3},
      {~w(list), &convert_term_list/3},
      {~w(list_cons), &convert_term_list_cons/3},
      {~w(list_flatten list_insert_at), &convert_term_read/3},
      {~w(map), &convert_term_map/3},
      {~w(
        map_put map_fetch mapset_from_list mapset_member mapset_put file_read file_read_lines binary_from_list
      ), &convert_term_read/3},
      {~w(binary), &convert_term_binary/3},
      {~w(
        iodata_to_binary float_lit bigint_lit string_to_float string_to_atom string_to_existing_atom
        float_to_binary_short
      ), &convert_term_read/3},
      {~w(is_integer is_float is_atom is_binary is_list is_tuple is_map), &convert_term_predicate/3},
      {~w(
        tuple_get tuple_length map_length enumerable_count enumerable_to_list enumerable_into_map
        enumerable_intersperse enumerable_to_list_range enumerable_reduce enumerable_reduce_c
        enumerable_reduce_range enumerable_reduce_fun enumerable_map_fun enumerable_map_term_fun
        enumerable_map_term_fun_c enumerable_flat_map_term_fun stream_filter stream_take stream_drop
      ), &convert_term_read/3},
      {~w(func_addr), &convert_func_addr/3},
      {~w(
        list_head list_tail list_get list_length term_eq term_eq_loose integer_compare integer_add integer_sub
        integer_mul integer_div integer_rem binary_length binary_get binary_slice binary_part binary_utf8_get
        binary_utf8_width binary_utf8_length string_printable binary_quote binary_encode16 binary_decode16
        int_to_string int_to_string_base int_to_hex string_to_int
      ), &convert_term_read/3}
    ]

    Enum.reduce(patterns, plan, fn {names, callback}, plan ->
      Enum.reduce(names, plan, fn name, plan ->
        Plan.add_conversion_pattern(plan, "ex." <> name, callback, version: "1.0")
      end)
    end)
  end

  defp populate_native_scalar_patterns(patterns, converter),
    do: CAPI.beaverPopulateExScalarConversionPatterns(patterns.ref, converter.ref)

  # Declaration-first manifest of the Zig term runtime ABI: batata's
  # `native/term_runtime.zig` exports exactly these C symbols.
  @intrinsic_names ~w(
    list_cons list_flatten list_insert_at map_put map_fetch self send receive mailbox_clear spawn runtime_create
    runtime_enter runtime_leave runtime_destroy result_create result_create_term result_destroy result_root_kind
    result_root_word result_exception_kind result_exception_reason result_term_kind result_atom_name
    result_term_length result_term_get term_export term_import exported_clone exported_destroy exported_length
    exported_get term_handle_export term_handle_destroy process_table_reset cont_save receive_cont_save
    cont_pending cont_active cont_clear cont_load_arg cont_load_acc cont_load_cursor schedule_next mailbox_len
    mailbox_peek mailbox_remove nil_word monotonic_time receive_start receive_start_set native_time unique_integer
    current_entry process_done process_exit process_exit_reason process_trap_exit process_dictionary_get
    process_dictionary_put link unlink exit monitor demonitor processes_runnable process_result worker_run
    process_wait to_int reduction_tick clock_init yield_mark jmp_buf_size setjmp_addr try_push try_pop throw raise
    catch_value make_fun make_fun_with_arity make_fun_with_signature fun_arity fun_result_mode fun_idx fun_env
    tuple_from_list map_from_list mapset_from_list mapset_member mapset_put file_read file_read_lines
    binary_from_list iodata_to_binary float_lit bigint_lit string_to_float string_to_atom string_to_existing_atom
    float_to_binary_short is_integer is_float is_atom is_binary is_list is_tuple is_map tuple_get tuple_length
    map_length enumerable_count enumerable_to_list enumerable_into_map enumerable_intersperse
    enumerable_to_list_range enumerable_reduce enumerable_reduce_c enumerable_reduce_range enumerable_reduce_fun
    enumerable_map_fun enumerable_map_term_fun enumerable_map_term_fun_c enumerable_flat_map_term_fun
    stream_filter stream_take stream_drop list_head list_tail list_get list_length term_eq term_eq_loose
    integer_compare integer_add integer_sub integer_mul integer_div integer_rem binary_length binary_get
    binary_slice binary_part binary_utf8_get binary_utf8_width binary_utf8_length string_printable binary_quote
    binary_encode16 binary_decode16 int_to_string int_to_string_base int_to_hex string_to_int
  )
  @intrinsic_aliases %{
    "term_export" => "export",
    "term_import" => "import",
    "term_handle_export" => "handle_export",
    "term_handle_destroy" => "handle_destroy",
    "nil_word" => "nil",
    "reduction_tick" => "clock_tick",
    "term_eq" => "eq",
    "term_eq_loose" => "eq_loose"
  }
  @term_intrinsics Map.new(@intrinsic_names, fn name ->
                     {String.to_atom(name), "ex.term." <> Map.get(@intrinsic_aliases, name, name)}
                   end)

  @doc """
  Returns the term runtime intrinsic symbols emitted by this conversion plan.

  This is the declaration-first anchor for host-boundary manifests (e.g.
  `Beaver.Wasm.TermABI`): every symbol here must be provided by the term
  runtime (batata's Zig on native, a wasm host import on wasm).
  """
  @spec term_intrinsic_symbols() :: [String.t()]
  def term_intrinsic_symbols, do: @term_intrinsics |> Map.values() |> Enum.sort()

  @doc """
  Converts an `ex` term type to its scalar word representation.
  """
  @spec convert_type(Type.t()) :: Type.t()
  def convert_type(type) do
    case MLIR.to_string(type) do
      "!ex.term" -> scalar_word(type)
      "!ex.bound" -> scalar_word(type)
      "!ex.unbound" -> scalar_word(type)
      _ -> type
    end
  end

  defp scalar_word(type) do
    ctx = MLIR.context(type)
    Type.integer(64, ctx: ctx)
  end

  defp convert_term_list(op, operands, rewriter) do
    base = insertion_point(op, rewriter)
    list = build_list(operands, op, rewriter, base)
    replace_with(rewriter, op, list)
  end

  defp convert_term_binary(op, operands, rewriter) do
    base = insertion_point(op, rewriter)
    list = build_list(operands, op, rewriter, base)

    binary = emit_runtime_call(op, rewriter, base, @term_intrinsics.binary_from_list, [list])

    replace_with(rewriter, op, binary)
  end

  defp convert_term_list_cons(op, [head, tail], rewriter), do: convert_term_read(op, [head, tail], rewriter)

  defp convert_term_tuple(op, operands, rewriter) do
    base = insertion_point(op, rewriter)
    list = build_list(operands, op, rewriter, base)

    tuple = emit_runtime_call(op, rewriter, base, @term_intrinsics.tuple_from_list, [list])

    replace_with(rewriter, op, tuple)
  end

  defp convert_term_map(op, operands, rewriter) do
    base = insertion_point(op, rewriter)

    if rem(length(operands), 2) != 0,
      do: raise(ArgumentError, "ex.map requires an even number of key/value operands, got #{length(operands)}")

    list = build_list(operands, op, rewriter, base)
    map = emit_runtime_call(op, rewriter, base, @term_intrinsics.map_from_list, [list])
    replace_with(rewriter, op, map)
  end

  defp convert_term_predicate(op, [operand], rewriter) do
    base = insertion_point(op, rewriter)

    word =
      case op |> Walker.operands() |> Enum.to_list() do
        [original] -> maybe_tag(original, operand, op, base)
        _ -> operand
      end

    intrinsic = op |> Operation.name() |> predicate_intrinsic()

    result = emit_runtime_call(op, rewriter, base, intrinsic, [word])
    replace_with(rewriter, op, result)
  end

  defp convert_self(op, [], rewriter), do: convert_term_read(op, [], rewriter)

  defp convert_send(op, [pid, msg], rewriter), do: convert_term_read(op, [pid, msg], rewriter)

  defp convert_receive(op, [], rewriter), do: convert_term_read(op, [], rewriter)

  defp convert_mailbox_clear(op, [], rewriter), do: convert_term_read(op, [], rewriter)

  defp convert_to_int(op, [operand], rewriter), do: convert_term_read(op, [operand], rewriter)

  # `ex.func_addr` resolves a function symbol to its address (an i64 word):
  # func.constant yields the function value, which func-to-llvm lowers to a
  # pointer. Used to hand a compiled reducer to the runtime's
  # arbitrary-closure enumerable reduce.
  defp convert_func_addr(op, [], rewriter) do
    context = MLIR.context(op)
    base = insertion_point(op, rewriter)
    location = Operation.location(op)
    {:ok, sym_name_attr} = op |> Operation.fetch(:sym_name)
    name = sym_name_attr |> CAPI.mlirStringAttrGetValue() |> MLIR.to_string()
    [result] = op |> Walker.results() |> Enum.to_list()
    result_type = result |> Value.type()

    func_const =
      %Changeset{name: "func.constant", context: context, location: location}
      |> Changeset.add_argument(value: Attribute.flat_symbol_ref(name, ctx: context))
      |> Changeset.add_result(result_type)
      |> Operation.create()

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
    location = Operation.location(op)
    [body_region, catch_region] = op |> Walker.regions() |> Enum.to_list()

    size = emit_runtime_call(op, rewriter, base, @term_intrinsics.jmp_buf_size, [])

    alloca =
      %Changeset{name: "llvm.alloca", context: context, location: location}
      |> Changeset.add_argument([size])
      |> Changeset.add_argument(
        elem_type: Attribute.type(Type.integer(8, ctx: context)),
        alignment: Attribute.integer(Type.i64(), 8)
      )
      |> Changeset.add_result(Type.llvm_pointer(ctx: context))
      |> Operation.create()

    RewriterBase.insert(base, alloca)
    buf = Operation.result(alloca, 0)

    push = emit_runtime_call(op, rewriter, base, @term_intrinsics.try_push, [buf])
    _ = push

    # libc's setjmp is resolved indirectly (the ORC linker does not resolve
    # libc symbols on its own): the runtime exposes its address.
    setjmp_addr = emit_runtime_call(op, rewriter, base, @term_intrinsics.setjmp_addr, [])

    fn_ptr =
      %Changeset{name: "llvm.inttoptr", context: context, location: location}
      |> Changeset.add_argument([setjmp_addr])
      |> Changeset.add_result(Type.llvm_pointer(ctx: context))
      |> Operation.create()

    RewriterBase.insert(base, fn_ptr)

    setjmp =
      %Changeset{name: "llvm.call", context: context, location: location}
      |> Changeset.add_argument([Operation.result(fn_ptr, 0), buf])
      |> Changeset.add_argument(
        operandSegmentSizes: Attribute.dense_array([2, 0], Native.I32, ctx: context),
        op_bundle_sizes: Attribute.dense_array([], Native.I32, ctx: context)
      )
      |> Changeset.add_result(Type.i64())
      |> Operation.create()

    RewriterBase.insert(base, setjmp)
    saved = Operation.result(setjmp, 0)

    zero = emit_constant(0, op, base) |> Operation.result(0)

    cmp =
      %Changeset{name: "arith.cmpi", context: context, location: location}
      |> Changeset.add_argument([saved, zero])
      |> Changeset.add_argument(predicate: cmp_i_predicate_attr(0))
      |> Changeset.add_result(Type.i1())
      |> Operation.create()

    RewriterBase.insert(base, cmp)

    scf_if =
      %Changeset{name: "scf.if", context: context, location: location}
      |> Changeset.add_argument(Operation.result(cmp, 0))
      |> Changeset.add_argument(CAPI.mlirRegionCreate())
      |> Changeset.add_argument(CAPI.mlirRegionCreate())
      |> Changeset.add_result([Type.i64()])
      |> Operation.create()

    [new_then, new_else] = scf_if |> Walker.regions() |> Enum.to_list()
    CAPI.mlirRegionTakeBody(new_then, body_region)
    CAPI.mlirRegionTakeBody(new_else, catch_region)

    prepend_try_pop(op, rewriter, new_else)
    append_try_pop(op, rewriter, new_then)

    RewriterBase.set_insertion_point_before(base, op)
    RewriterBase.insert(base, scf_if)

    ConversionPatternRewriter.replace_op(rewriter, op, scf_if |> Walker.results() |> Enum.to_list())

    :ok
  end

  # The catch path pops the try buffer first: the longjmp returns with the
  # buffer still pushed.
  defp prepend_try_pop(op, rewriter, region) do
    ensure_intrinsic_declaration(op, rewriter, @term_intrinsics.try_pop)
    [block] = region |> Walker.blocks() |> Enum.to_list()
    base = ConversionPatternRewriter.as_base(rewriter)
    RewriterBase.set_insertion_point_to_start(base, block)
    emit_try_pop_call(op, rewriter)
    :ok
  end

  # The normal path pops the try buffer before leaving the body.
  defp append_try_pop(op, rewriter, region) do
    ensure_intrinsic_declaration(op, rewriter, @term_intrinsics.try_pop)
    [block] = region |> Walker.blocks() |> Enum.to_list()
    terminator = CAPI.mlirBlockGetTerminator(block)
    base = ConversionPatternRewriter.as_base(rewriter)
    RewriterBase.set_insertion_point_before(base, terminator)
    emit_try_pop_call(op, rewriter)
    :ok
  end

  # Builds a func.call to the try_pop intrinsic at the current insertion
  # point (unlike `emit_runtime_call`, this does not move the insertion
  # point, so it can be placed inside a region).
  defp emit_try_pop_call(op, rewriter) do
    base = ConversionPatternRewriter.as_base(rewriter)
    symbol = @term_intrinsics.try_pop

    call =
      %Changeset{name: "func.call", context: MLIR.context(op), location: Operation.location(op)}
      |> Changeset.add_argument([])
      |> Changeset.add_argument(callee: Attribute.flat_symbol_ref(symbol, ctx: MLIR.context(op)))
      |> Changeset.add_result(Type.i64())
      |> Operation.create()

    RewriterBase.insert(base, call)
    Operation.result(call, 0)
  end

  defp convert_throw(op, [operand], rewriter), do: convert_term_read(op, [operand], rewriter)

  defp convert_raise(op, [reason, kind], rewriter), do: convert_term_read(op, [reason, kind], rewriter)

  defp convert_catch_value(op, [], rewriter), do: convert_term_read(op, [], rewriter)

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
    ctx = MLIR.context(op)
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

    dispatch =
      %Changeset{name: "func.call", context: ctx, location: Operation.location(op)}
      |> Changeset.add_argument([idx] ++ envs ++ args ++ pad)
      |> Changeset.add_argument(callee: Attribute.flat_symbol_ref("__fn_dispatch", ctx: ctx))
      |> Changeset.add_result(Type.i64())
      |> Operation.create()

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

  defp predicate_intrinsic(name), do: read_intrinsic(name)

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

    shl =
      %Changeset{name: "arith.shli", context: MLIR.context(op), location: Operation.location(op)}
      |> Changeset.add_argument([value, Operation.result(three, 0)])
      |> Changeset.add_result(Type.i64())
      |> Operation.create()

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

  defp emit_constant(value, op, base) do
    constant =
      %Changeset{
        name: "arith.constant",
        context: MLIR.context(op),
        location: Operation.location(op)
      }
      |> Changeset.add_argument(value: Attribute.integer(Type.i64(), value))
      |> Changeset.add_result(Type.i64())
      |> Operation.create()

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

    call =
      %Changeset{name: "func.call", context: MLIR.context(op), location: Operation.location(op)}
      |> Changeset.add_argument(args)
      |> Changeset.add_argument(callee: Attribute.flat_symbol_ref(symbol, ctx: MLIR.context(op)))
      |> Changeset.add_result(Type.i64())
      |> Operation.create()

    RewriterBase.insert(base, call)
    RewriterBase.set_insertion_point_after(base, call)
    Operation.result(call, 0)
  end

  # `func.call` requires the callee symbol to exist, so each intrinsic gets a
  # public `func.func` declaration (no body) at module scope. At link time the
  # Zig term runtime shared library satisfies the symbols.
  defp ensure_intrinsic_declaration(operation, rewriter, symbol) do
    module_op =
      Stream.iterate(operation, &Operation.parent/1)
      |> Enum.find(fn op -> op |> Operation.parent() |> MLIR.null?() end)

    body = module_body(module_op)

    unless declaration_exists?(body, symbol) do
      base = ConversionPatternRewriter.as_base(rewriter)
      RewriterBase.set_insertion_point_to_end(base, body)

      declaration =
        %Changeset{name: "func.func", context: MLIR.context(operation), location: Operation.location(operation)}
        |> Changeset.add_argument(sym_name: Attribute.string(symbol))
        |> Changeset.add_argument(sym_visibility: Attribute.string("private"))
        |> Changeset.add_argument(function_type: intrinsic_function_type(symbol, MLIR.context(operation)))
        |> Changeset.add_argument(CAPI.mlirRegionCreate())
        |> Operation.create()

      RewriterBase.insert(base, declaration)
    end

    :ok
  end

  defp declaration_exists?(body, symbol) do
    body
    |> Walker.operations()
    |> Enum.to_list()
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

  # Most runtime functions consume and return tagged i64 words. Pointer and
  # callback signatures below stay explicit so their ABI remains inspectable.
  for {arity, names} <- [
        {0, ~w(
          self receive yield_mark mailbox_clear runtime_create runtime_leave cont_pending cont_active cont_clear
          cont_load_arg cont_load_acc cont_load_cursor schedule_next mailbox_len nil monotonic_time receive_start
          native_time current_entry processes_runnable jmp_buf_size setjmp_addr try_pop catch_value
        )},
        {2, ~w(
          list_cons send raise enumerable_to_list_range fun_env tuple_get list_get eq eq_loose integer_compare
          integer_add integer_sub integer_mul integer_div integer_rem binary_get binary_slice binary_utf8_get
          binary_utf8_width int_to_string_base enumerable_into_map enumerable_intersperse map_fetch mapset_member
          mapset_put stream_take stream_drop result_create result_create_term result_term_kind result_atom_name
          result_term_length export import exported_get process_dictionary_get process_dictionary_put
        )},
        {3, ~w(map_put list_insert_at result_term_get cont_save receive_cont_save link enumerable_reduce binary_part)},
        {4, ~w(exit monitor enumerable_reduce_c enumerable_reduce_range)},
        {6, ~w(make_fun)},
        {7, ~w(make_fun_with_arity)},
        {8, ~w(make_fun_with_signature)}
      ],
      name <- names do
    defp intrinsic_function_type(unquote("ex.term." <> name), _ctx),
      do: Type.function(List.duplicate(Type.i64(), unquote(arity)), [Type.i64()])
  end

  defp intrinsic_function_type("ex.term.try_push", ctx), do: Type.function([Type.llvm_pointer(ctx: ctx)], [Type.i64()])

  defp intrinsic_function_type("ex.term.enumerable_reduce_fun", _ctx),
    do: Type.function([Type.i64(), Type.i64(), Type.function([Type.i64(), Type.i64()], [Type.i64()])], [Type.i64()])

  defp intrinsic_function_type("ex.term.worker_run", _ctx),
    do: Type.function([Type.i64(), Type.function([Type.i64()], [Type.i64()])], [Type.i64()])

  defp intrinsic_function_type(symbol, _ctx)
       when symbol in ~w(
         ex.term.enumerable_map_fun ex.term.enumerable_map_term_fun ex.term.enumerable_flat_map_term_fun
         ex.term.stream_filter
       ) do
    Type.function([Type.i64(), Type.function([Type.i64()], [Type.i64()])], [Type.i64()])
  end

  defp intrinsic_function_type("ex.term.enumerable_map_term_fun_c", _ctx) do
    callback = Type.function(List.duplicate(Type.i64(), 8), [Type.i64()])

    Type.function([Type.i64(), callback | List.duplicate(Type.i64(), 4)], [Type.i64()])
  end

  defp intrinsic_function_type(_symbol, _ctx), do: Type.function([Type.i64()], [Type.i64()])

  defp term_type?(type), do: MLIR.to_string(type) in @term_types

  defp cmp_i_predicate_attr(i), do: Attribute.integer(Type.i64(), i)

  defp convert_call(op, args, rewriter) do
    context = MLIR.context(op)
    base = ConversionPatternRewriter.as_base(rewriter)
    location = Operation.location(op)

    callee = op |> required_attribute("callee") |> CAPI.mlirStringAttrGetValue() |> MLIR.to_string()

    arity = op |> required_attribute("arity") |> CAPI.mlirIntegerAttrGetValueInt() |> Native.to_term()

    unless length(args) == arity,
      do: raise(ArgumentError, "ex.call arity attribute #{arity} does not match #{length(args)} arguments")

    [result] = op |> Walker.results() |> Enum.to_list()
    result_type = result |> Value.type() |> convert_type()

    call =
      %Changeset{name: "func.call", context: context, location: location}
      |> Changeset.add_argument(args)
      |> Changeset.add_argument(callee: Attribute.flat_symbol_ref(callee, ctx: context))
      |> Changeset.add_result(result_type)
      |> Operation.create()

    RewriterBase.set_insertion_point_before(base, op)
    RewriterBase.insert(base, call)
    ConversionPatternRewriter.replace_op(rewriter, op, Operation.result(call, 0))
    :ok
  end

  defp convert_return(op, operands, rewriter) do
    context = MLIR.context(op)
    base = ConversionPatternRewriter.as_base(rewriter)
    location = Operation.location(op)

    return_op =
      %Changeset{name: "func.return", context: context, location: location}
      |> Changeset.add_argument(operands)
      |> Operation.create()

    RewriterBase.set_insertion_point_before(base, op)
    RewriterBase.insert(base, return_op)
    ConversionPatternRewriter.replace_op(rewriter, op, return_op)
    :ok
  end

  defp convert_var(op, [], _rewriter),
    do: raise(ArgumentError, "ex.var without ex.bind is unsupported: #{inspect(MLIR.to_string(op))}")

  defp convert_func(op, [], rewriter) do
    context = MLIR.context(op)
    base = ConversionPatternRewriter.as_base(rewriter)
    location = Operation.location(op)

    sym_name = op |> required_attribute("sym_name") |> CAPI.mlirStringAttrGetValue() |> MLIR.to_string()

    [body_region] = op |> Walker.regions() |> Enum.to_list()
    [block] = body_region |> Walker.blocks() |> Enum.to_list()
    terminator = CAPI.mlirBlockGetTerminator(block)

    if MLIR.null?(terminator), do: raise(ArgumentError, "ex.func body must end with a terminator (ex.return)")

    return_types =
      terminator |> Walker.operands() |> Enum.to_list() |> Enum.map(&(&1 |> Value.type() |> convert_type()))

    arg_types = block |> Walker.arguments() |> Enum.to_list() |> Enum.map(&(&1 |> Value.type() |> convert_type()))

    function_type = Type.function(arg_types, return_types)

    func =
      %Changeset{name: "func.func", context: context, location: location}
      |> Changeset.add_argument(sym_name: Attribute.string(sym_name))
      |> Changeset.add_argument(function_type: function_type)
      |> Changeset.add_argument(CAPI.mlirRegionCreate())
      |> Operation.create()

    func_body = func |> Walker.regions() |> Enum.to_list() |> hd()
    CAPI.mlirRegionTakeBody(func_body, body_region)

    RewriterBase.set_insertion_point_before(base, op)
    RewriterBase.insert(base, func)
    ConversionPatternRewriter.replace_op(rewriter, op, func)
    :ok
  end

  defp required_attribute(op, name) do
    case op |> Walker.attributes() |> then(& &1[name]) do
      nil ->
        raise ArgumentError,
              "#{Operation.name(op)} requires attribute #{inspect(name)}"

      attribute ->
        attribute
    end
  end
end
