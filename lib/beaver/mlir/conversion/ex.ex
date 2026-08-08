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

  @doc """
  Returns a conversion plan lowering the `ex` scalar subset to `func`/`arith`.
  """
  @spec plan(keyword()) :: Plan.t()
  def plan(opts \\ []) do
    Plan.new(
      Keyword.merge(
        [
          mode: :full,
          folding_mode: :after_patterns,
          build_materializations: true
        ],
        opts
      )
    )
    |> Plan.add_legal_dialect("builtin")
    |> Plan.add_legal_dialect("func")
    |> Plan.add_legal_dialect("arith")
    |> Plan.add_legal_dialect("cf")
    |> Plan.add_legal_dialect("scf")
    |> Plan.add_illegal_dialect("ex")
    |> Plan.add_conversion(&convert_type/1, version: "1.0")
    |> Plan.add_conversion_pattern("ex.lit", &convert_lit/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.add", &convert_add/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.sub", &convert_sub/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mul", &convert_mul/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cmp", &convert_cmp/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.if", &convert_if/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.yield", &convert_yield/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.call", &convert_call/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.return", &convert_return/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.var", &convert_var/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.func", &convert_func/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.box", &convert_box/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.to_word", &convert_to_word/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.self", &convert_self/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.send", &convert_send/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.receive", &convert_receive/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mailbox_clear", &convert_mailbox_clear/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.to_int", &convert_to_int/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.make_fun", &convert_make_fun/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.apply", &convert_apply/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.tuple", &convert_term_tuple/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list", &convert_term_list/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.map", &convert_term_map/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary", &convert_term_binary/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_integer", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_atom", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_binary", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_list", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_tuple", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_map", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.tuple_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.tuple_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list_head", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list_tail", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.term_eq", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_slice", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_utf8_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_utf8_width", &convert_term_read/3, version: "1.0")
  end

  # Declaration-first manifest of the Zig term runtime ABI: batata's
  # `native/term_runtime.zig` exports exactly these C symbols.
  @term_intrinsics %{
    list_cons: "ex.term.list_cons",
    self: "ex.term.self",
    send: "ex.term.send",
    receive: "ex.term.receive",
    mailbox_clear: "ex.term.mailbox_clear",
    to_int: "ex.term.to_int",
    make_fun: "ex.term.make_fun",
    fun_idx: "ex.term.fun_idx",
    fun_env: "ex.term.fun_env",
    tuple_from_list: "ex.term.tuple_from_list",
    map_from_list: "ex.term.map_from_list",
    binary_from_list: "ex.term.binary_from_list",
    is_integer: "ex.term.is_integer",
    is_atom: "ex.term.is_atom",
    is_binary: "ex.term.is_binary",
    is_list: "ex.term.is_list",
    is_tuple: "ex.term.is_tuple",
    is_map: "ex.term.is_map",
    tuple_get: "ex.term.tuple_get",
    tuple_length: "ex.term.tuple_length",
    list_head: "ex.term.list_head",
    list_tail: "ex.term.list_tail",
    list_length: "ex.term.list_length",
    term_eq: "ex.term.eq",
    binary_length: "ex.term.binary_length",
    binary_get: "ex.term.binary_get",
    binary_slice: "ex.term.binary_slice",
    binary_utf8_get: "ex.term.binary_utf8_get",
    binary_utf8_width: "ex.term.binary_utf8_width"
  }

  @term_types ~w(!ex.dyn !ex.bound !ex.unbound)

  @doc """
  Converts an `ex` term type to its scalar word representation.
  """
  @spec convert_type(MLIR.Type.t()) :: MLIR.Type.t()
  def convert_type(type) do
    case MLIR.to_string(type) do
      "!ex.dyn" -> scalar_word(type)
      "!ex.bound" -> scalar_word(type)
      "!ex.unbound" -> scalar_word(type)
      _ -> type
    end
  end

  defp scalar_word(type) do
    ctx = MLIR.context(type)
    MLIR.Type.integer(64, ctx: ctx)
  end

  defp convert_lit(operation, [], rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)
    [result] = operation |> Walker.results() |> Enum.to_list()
    result_type = result |> MLIR.Value.type() |> convert_type()

    constant =
      %Changeset{name: "arith.constant", context: context, location: location}
      |> Changeset.add_argument(value: required_attribute(operation, "value"))
      |> Changeset.add_result(result_type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, constant)

    MLIR.ConversionPatternRewriter.replace_op(
      rewriter,
      operation,
      MLIR.Operation.result(constant, 0)
    )

    :ok
  end

  defp convert_add(operation, [left, right], rewriter) do
    convert_binary("arith.addi", operation, [left, right], rewriter)
  end

  defp convert_sub(operation, [left, right], rewriter) do
    convert_binary("arith.subi", operation, [left, right], rewriter)
  end

  defp convert_mul(operation, [left, right], rewriter) do
    convert_binary("arith.muli", operation, [left, right], rewriter)
  end

  defp convert_binary(arith_op, operation, [left, right], rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)
    [result] = operation |> Walker.results() |> Enum.to_list()
    result_type = result |> MLIR.Value.type() |> convert_type()

    add =
      %Changeset{name: arith_op, context: context, location: location}
      |> Changeset.add_argument([left, right])
      |> Changeset.add_result(result_type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, add)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, MLIR.Operation.result(add, 0))
    :ok
  end

  defp convert_cmp(operation, [left, right], rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)
    [result] = operation |> Walker.results() |> Enum.to_list()
    result_type = result |> MLIR.Value.type() |> convert_type()

    predicate =
      operation
      |> required_attribute("predicate")
      |> MLIR.CAPI.mlirStringAttrGetValue()
      |> MLIR.to_string()

    cmpi =
      %Changeset{name: "arith.cmpi", context: context, location: location}
      |> Changeset.add_argument([left, right])
      |> Changeset.add_argument(predicate: cmp_i_predicate(predicate))
      |> Changeset.add_result(MLIR.Type.i1())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, cmpi)

    # Comparisons produce i1 in arith; widen back to the ex.cmp result's i64
    # boolean representation so it can feed arithmetic and scf.if conditions.
    ext =
      %Changeset{name: "arith.extui", context: context, location: location}
      |> Changeset.add_argument([MLIR.Operation.result(cmpi, 0)])
      |> Changeset.add_result(result_type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_after(base, cmpi)
    MLIR.RewriterBase.insert(base, ext)

    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, MLIR.Operation.result(ext, 0))
    :ok
  end

  defp convert_if(operation, [cond], rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)

    result_types =
      operation
      |> Walker.results()
      |> Enum.to_list()
      |> Enum.map(&(&1 |> MLIR.Value.type() |> convert_type()))

    [then_region, else_region] = operation |> Walker.regions() |> Enum.to_list()

    # scf.if conditions are i1 at the LLVM boundary; the ex universe carries
    # booleans as i64 0/1, so truncate before the branch.
    cond_i1 =
      %Changeset{name: "arith.trunci", context: context, location: location}
      |> Changeset.add_argument([cond])
      |> Changeset.add_result(MLIR.Type.i1())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, cond_i1)

    scf_if =
      %Changeset{name: "scf.if", context: context, location: location}
      |> Changeset.add_argument(MLIR.Operation.result(cond_i1, 0))
      |> Changeset.add_argument(MLIR.CAPI.mlirRegionCreate())
      |> Changeset.add_argument(MLIR.CAPI.mlirRegionCreate())
      |> Changeset.add_result(result_types)
      |> MLIR.Operation.create()

    [new_then, new_else] = scf_if |> Walker.regions() |> Enum.to_list()
    MLIR.CAPI.mlirRegionTakeBody(new_then, then_region)
    MLIR.CAPI.mlirRegionTakeBody(new_else, else_region)

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, scf_if)

    MLIR.ConversionPatternRewriter.replace_op(
      rewriter,
      operation,
      scf_if |> Walker.results() |> Enum.to_list()
    )

    :ok
  end

  defp convert_yield(operation, operands, rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)

    yield =
      %Changeset{name: "scf.yield", context: context, location: location}
      |> Changeset.add_argument(operands)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, yield)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, yield)
    :ok
  end

  defp convert_term_list(operation, operands, rewriter) do
    base = insertion_point(operation, rewriter)
    list = build_list(operands, operation, rewriter, base)
    replace_with(rewriter, operation, list)
  end

  defp convert_term_tuple(operation, operands, rewriter) do
    base = insertion_point(operation, rewriter)
    list = build_list(operands, operation, rewriter, base)

    tuple = emit_runtime_call(operation, rewriter, base, @term_intrinsics.tuple_from_list, [list])

    replace_with(rewriter, operation, tuple)
  end

  defp convert_term_map(operation, operands, rewriter) do
    base = insertion_point(operation, rewriter)

    if rem(length(operands), 2) != 0 do
      raise ArgumentError,
            "ex.map requires an even number of key/value operands, got #{length(operands)}"
    end

    list = build_list(operands, operation, rewriter, base)
    map = emit_runtime_call(operation, rewriter, base, @term_intrinsics.map_from_list, [list])
    replace_with(rewriter, operation, map)
  end

  defp convert_term_binary(operation, operands, rewriter) do
    base = insertion_point(operation, rewriter)
    list = build_list(operands, operation, rewriter, base)

    binary =
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.binary_from_list, [list])

    replace_with(rewriter, operation, binary)
  end

  defp convert_term_predicate(operation, [operand], rewriter) do
    base = insertion_point(operation, rewriter)

    word =
      case operation |> Walker.operands() |> Enum.to_list() do
        [original] -> maybe_tag(original, operand, operation, base)
        _ -> operand
      end

    intrinsic =
      operation
      |> MLIR.Operation.name()
      |> predicate_intrinsic()

    result = emit_runtime_call(operation, rewriter, base, intrinsic, [word])
    replace_with(rewriter, operation, result)
  end

  defp convert_box(operation, [operand], rewriter) do
    base = insertion_point(operation, rewriter)

    word =
      case operation |> Walker.operands() |> Enum.to_list() do
        [original] -> maybe_tag(original, operand, operation, base)
        _ -> operand
      end

    replace_with(rewriter, operation, word)
  end

  defp convert_to_word(operation, [operand], rewriter) do
    replace_with(rewriter, operation, operand)
  end

  defp convert_self(operation, [], rewriter) do
    base = insertion_point(operation, rewriter)
    replace_with(rewriter, operation, emit_runtime_call(operation, rewriter, base, @term_intrinsics.self, []))
  end

  defp convert_send(operation, [pid, msg], rewriter) do
    base = insertion_point(operation, rewriter)

    replace_with(
      rewriter,
      operation,
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.send, [pid, msg])
    )
  end

  defp convert_receive(operation, [], rewriter) do
    base = insertion_point(operation, rewriter)

    replace_with(
      rewriter,
      operation,
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.receive, [])
    )
  end

  defp convert_mailbox_clear(operation, [], rewriter) do
    base = insertion_point(operation, rewriter)

    replace_with(
      rewriter,
      operation,
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.mailbox_clear, [])
    )
  end

  defp convert_to_int(operation, [operand], rewriter) do
    base = insertion_point(operation, rewriter)

    replace_with(
      rewriter,
      operation,
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.to_int, [operand])
    )
  end

  # Constructs a first-class function value: stores the function index and the
  # captured env words in a closure word allocated by the Zig runtime.
  defp convert_make_fun(operation, operands, rewriter) do
    base = insertion_point(operation, rewriter)

    fn_idx = operation |> required_attribute("fn_idx") |> MLIR.CAPI.mlirIntegerAttrGetValueInt()
    env_len = operation |> required_attribute("env_len") |> MLIR.CAPI.mlirIntegerAttrGetValueInt()
    env_len = Beaver.Native.to_term(env_len)

    unless length(operands) == env_len do
      raise ArgumentError,
            "ex.make_fun env_len #{env_len} does not match #{length(operands)} operands"
    end

    idx_const =
      emit_constant(Beaver.Native.to_term(fn_idx), operation, base) |> MLIR.Operation.result(0)

    len_const = emit_constant(env_len, operation, base) |> MLIR.Operation.result(0)

    pad =
      List.duplicate(emit_constant(0, operation, base) |> MLIR.Operation.result(0), 4 - env_len)

    closure =
      emit_runtime_call(
        operation,
        rewriter,
        base,
        @term_intrinsics.make_fun,
        [idx_const, len_const] ++ operands ++ pad
      )

    replace_with(rewriter, operation, closure)
  end

  # Applies a first-class function value: reads the function index and env
  # words from the closure, then dispatches to the matching `__fn_*`.
  defp convert_apply(operation, operands, rewriter) do
    base = insertion_point(operation, rewriter)
    ctx = MLIR.context(operation)
    [closure | args] = operands

    arg_count =
      operation
      |> required_attribute("arg_count")
      |> MLIR.CAPI.mlirIntegerAttrGetValueInt()
      |> Beaver.Native.to_term()

    unless length(args) == arg_count do
      raise ArgumentError,
            "ex.apply arg_count #{arg_count} does not match #{length(args)} arguments"
    end

    idx = emit_runtime_call(operation, rewriter, base, @term_intrinsics.fun_idx, [closure])

    envs =
      for i <- 0..3 do
        index_const = emit_constant(i, operation, base) |> MLIR.Operation.result(0)

        emit_runtime_call(operation, rewriter, base, @term_intrinsics.fun_env, [
          closure,
          index_const
        ])
      end

    pad =
      List.duplicate(emit_constant(0, operation, base) |> MLIR.Operation.result(0), 4 - arg_count)

    dispatch =
      %Changeset{
        name: "func.call",
        context: ctx,
        location: MLIR.Operation.location(operation)
      }
      |> Changeset.add_argument([idx] ++ envs ++ args ++ pad)
      |> Changeset.add_argument(callee: MLIR.Attribute.flat_symbol_ref("__fn_dispatch", ctx: ctx))
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, dispatch)
    MLIR.RewriterBase.set_insertion_point_after(base, dispatch)
    replace_with(rewriter, operation, MLIR.Operation.result(dispatch, 0))
  end

  defp predicate_intrinsic("ex.is_integer"), do: @term_intrinsics.is_integer
  defp predicate_intrinsic("ex.is_atom"), do: @term_intrinsics.is_atom
  defp predicate_intrinsic("ex.is_binary"), do: @term_intrinsics.is_binary
  defp predicate_intrinsic("ex.is_list"), do: @term_intrinsics.is_list
  defp predicate_intrinsic("ex.is_tuple"), do: @term_intrinsics.is_tuple
  defp predicate_intrinsic("ex.is_map"), do: @term_intrinsics.is_map

  defp convert_term_read(operation, operands, rewriter) do
    base = insertion_point(operation, rewriter)
    symbol = read_intrinsic(MLIR.Operation.name(operation))
    result = emit_runtime_call(operation, rewriter, base, symbol, operands)
    replace_with(rewriter, operation, result)
  end

  defp read_intrinsic("ex.tuple_get"), do: @term_intrinsics.tuple_get
  defp read_intrinsic("ex.tuple_length"), do: @term_intrinsics.tuple_length
  defp read_intrinsic("ex.list_head"), do: @term_intrinsics.list_head
  defp read_intrinsic("ex.list_tail"), do: @term_intrinsics.list_tail
  defp read_intrinsic("ex.list_length"), do: @term_intrinsics.list_length
  defp read_intrinsic("ex.term_eq"), do: @term_intrinsics.term_eq
  defp read_intrinsic("ex.binary_length"), do: @term_intrinsics.binary_length
  defp read_intrinsic("ex.binary_get"), do: @term_intrinsics.binary_get
  defp read_intrinsic("ex.binary_slice"), do: @term_intrinsics.binary_slice
  defp read_intrinsic("ex.binary_utf8_get"), do: @term_intrinsics.binary_utf8_get
  defp read_intrinsic("ex.binary_utf8_width"), do: @term_intrinsics.binary_utf8_width

  defp insertion_point(operation, rewriter) do
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    base
  end

  defp replace_with(rewriter, operation, value) do
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, value)
    :ok
  end

  defp maybe_tag(original, converted, operation, base) do
    if term_type?(MLIR.Value.type(original)) do
      converted
    else
      tag_integer(converted, operation, base)
    end
  end

  # Tags a scalar i64 as an immediate integer term: word = value << 3 with the
  # low 3 bits (tag 0b000) left as the integer tag.
  defp tag_integer(value, operation, base) do
    three = emit_constant(3, operation, base)

    shl =
      %Changeset{
        name: "arith.shli",
        context: MLIR.context(operation),
        location: MLIR.Operation.location(operation)
      }
      |> Changeset.add_argument([value, MLIR.Operation.result(three, 0)])
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, shl)
    MLIR.RewriterBase.set_insertion_point_after(base, shl)
    MLIR.Operation.result(shl, 0)
  end

  # Builds a proper list word by consing words onto nil. Nil is the atom term
  # with id 0: tag 0b001 | (0 << 3) = 1.
  defp build_list([], operation, _rewriter, base) do
    emit_constant(1, operation, base) |> MLIR.Operation.result(0)
  end

  defp build_list(words, operation, rewriter, base) do
    words
    |> Enum.reverse()
    |> Enum.reduce(emit_constant(1, operation, base) |> MLIR.Operation.result(0), fn
      word, tail ->
        emit_runtime_call(operation, rewriter, base, @term_intrinsics.list_cons, [word, tail])
    end)
  end

  defp emit_constant(value, operation, base) do
    constant =
      %Changeset{
        name: "arith.constant",
        context: MLIR.context(operation),
        location: MLIR.Operation.location(operation)
      }
      |> Changeset.add_argument(value: MLIR.Attribute.integer(MLIR.Type.i64(), value))
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, constant)
    MLIR.RewriterBase.set_insertion_point_after(base, constant)
    constant
  end

  defp emit_runtime_call(operation, rewriter, base, symbol, args) do
    ensure_intrinsic_declaration(operation, rewriter, symbol)
    MLIR.RewriterBase.set_insertion_point_before(base, operation)

    call =
      %Changeset{
        name: "func.call",
        context: MLIR.context(operation),
        location: MLIR.Operation.location(operation)
      }
      |> Changeset.add_argument(args)
      |> Changeset.add_argument(
        callee: MLIR.Attribute.flat_symbol_ref(symbol, ctx: MLIR.context(operation))
      )
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, call)
    MLIR.RewriterBase.set_insertion_point_after(base, call)
    MLIR.Operation.result(call, 0)
  end

  # `func.call` requires the callee symbol to exist, so each intrinsic gets a
  # public `func.func` declaration (no body) at module scope. At link time the
  # Zig term runtime shared library satisfies the symbols.
  defp ensure_intrinsic_declaration(operation, rewriter, symbol) do
    module_op =
      Stream.iterate(operation, &MLIR.Operation.parent/1)
      |> Enum.find(fn op -> op |> MLIR.Operation.parent() |> MLIR.null?() end)

    body = module_body(module_op)

    unless declaration_exists?(body, symbol) do
      base = MLIR.ConversionPatternRewriter.as_base(rewriter)
      MLIR.RewriterBase.set_insertion_point_to_end(base, body)

      declaration =
        %Changeset{
          name: "func.func",
          context: MLIR.context(operation),
          location: MLIR.Operation.location(operation)
        }
        |> Changeset.add_argument(sym_name: MLIR.Attribute.string(symbol))
        |> Changeset.add_argument(sym_visibility: MLIR.Attribute.string("private"))
        |> Changeset.add_argument(function_type: intrinsic_function_type(symbol))
        |> Changeset.add_argument(MLIR.CAPI.mlirRegionCreate())
        |> MLIR.Operation.create()

      MLIR.RewriterBase.insert(base, declaration)
    end

    :ok
  end

  defp declaration_exists?(body, symbol) do
    body
    |> Walker.operations()
    |> Enum.to_list()
    |> Enum.any?(fn op ->
      MLIR.Operation.name(op) == "func.func" and
        case op |> MLIR.Operation.fetch("sym_name") do
          {:ok, attribute} ->
            attribute |> MLIR.CAPI.mlirStringAttrGetValue() |> MLIR.to_string() == symbol

          :error ->
            false
        end
    end)
  end

  defp module_body(module_op) do
    module_op
    |> Walker.regions()
    |> Enum.to_list()
    |> hd()
    |> Walker.blocks()
    |> Enum.to_list()
    |> hd()
  end

  defp intrinsic_function_type("ex.term.list_cons") do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.self") do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.send") do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.receive") do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.mailbox_clear") do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.make_fun") do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 6), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.fun_env") do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(symbol)
       when symbol in [
              "ex.term.tuple_get",
              "ex.term.eq",
              "ex.term.binary_get",
              "ex.term.binary_slice",
              "ex.term.binary_utf8_get",
              "ex.term.binary_utf8_width"
            ] do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(_symbol) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp term_type?(type), do: MLIR.to_string(type) in @term_types

  defp cmp_i_predicate("eq"), do: cmp_i_predicate_attr(0)
  defp cmp_i_predicate("ne"), do: cmp_i_predicate_attr(1)
  defp cmp_i_predicate("slt"), do: cmp_i_predicate_attr(2)
  defp cmp_i_predicate("sle"), do: cmp_i_predicate_attr(3)
  defp cmp_i_predicate("sgt"), do: cmp_i_predicate_attr(4)
  defp cmp_i_predicate("sge"), do: cmp_i_predicate_attr(5)
  defp cmp_i_predicate("ult"), do: cmp_i_predicate_attr(6)
  defp cmp_i_predicate("ule"), do: cmp_i_predicate_attr(7)
  defp cmp_i_predicate("ugt"), do: cmp_i_predicate_attr(8)
  defp cmp_i_predicate("uge"), do: cmp_i_predicate_attr(9)

  defp cmp_i_predicate(other) do
    raise ArgumentError, "unsupported ex.cmp predicate: #{inspect(other)}"
  end

  defp cmp_i_predicate_attr(i) do
    MLIR.Attribute.integer(MLIR.Type.i64(), i)
  end

  defp convert_call(operation, args, rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)

    callee =
      operation
      |> required_attribute("callee")
      |> MLIR.CAPI.mlirStringAttrGetValue()
      |> MLIR.to_string()

    arity =
      operation
      |> required_attribute("arity")
      |> MLIR.CAPI.mlirIntegerAttrGetValueInt()
      |> Beaver.Native.to_term()

    unless length(args) == arity do
      raise ArgumentError,
            "ex.call arity attribute #{arity} does not match #{length(args)} arguments"
    end

    [result] = operation |> Walker.results() |> Enum.to_list()
    result_type = result |> MLIR.Value.type() |> convert_type()

    call =
      %Changeset{name: "func.call", context: context, location: location}
      |> Changeset.add_argument(args)
      |> Changeset.add_argument(callee: MLIR.Attribute.flat_symbol_ref(callee, ctx: context))
      |> Changeset.add_result(result_type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, call)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, MLIR.Operation.result(call, 0))
    :ok
  end

  defp convert_return(operation, operands, rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)

    return_op =
      %Changeset{name: "func.return", context: context, location: location}
      |> Changeset.add_argument(operands)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, return_op)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, return_op)
    :ok
  end

  defp convert_var(operation, [], _rewriter) do
    raise ArgumentError,
          "ex.var without ex.bind is unsupported: #{inspect(MLIR.to_string(operation))}"
  end

  defp convert_func(operation, [], rewriter) do
    context = MLIR.context(operation)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    location = MLIR.Operation.location(operation)

    sym_name =
      operation
      |> required_attribute("sym_name")
      |> MLIR.CAPI.mlirStringAttrGetValue()
      |> MLIR.to_string()

    [body_region] = operation |> Walker.regions() |> Enum.to_list()
    [block] = body_region |> Walker.blocks() |> Enum.to_list()
    terminator = MLIR.CAPI.mlirBlockGetTerminator(block)

    if MLIR.null?(terminator),
      do: raise(ArgumentError, "ex.func body must end with a terminator (ex.return)")

    return_types =
      terminator
      |> Walker.operands()
      |> Enum.to_list()
      |> Enum.map(&(&1 |> MLIR.Value.type() |> convert_type()))

    arg_types =
      block
      |> Walker.arguments()
      |> Enum.to_list()
      |> Enum.map(&(&1 |> MLIR.Value.type() |> convert_type()))

    function_type = MLIR.Type.function(arg_types, return_types)

    func =
      %Changeset{name: "func.func", context: context, location: location}
      |> Changeset.add_argument(sym_name: MLIR.Attribute.string(sym_name))
      |> Changeset.add_argument(function_type: function_type)
      |> Changeset.add_argument(MLIR.CAPI.mlirRegionCreate())
      |> MLIR.Operation.create()

    func_body = func |> Walker.regions() |> Enum.to_list() |> hd()
    MLIR.CAPI.mlirRegionTakeBody(func_body, body_region)

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, func)
    MLIR.ConversionPatternRewriter.replace_op(rewriter, operation, func)
    :ok
  end

  defp required_attribute(operation, name) do
    case operation |> Walker.attributes() |> then(& &1[name]) do
      nil ->
        raise ArgumentError,
              "#{MLIR.Operation.name(operation)} requires attribute #{inspect(name)}"

      attribute ->
        attribute
    end
  end
end
