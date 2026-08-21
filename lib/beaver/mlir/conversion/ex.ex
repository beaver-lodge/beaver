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
    |> Plan.add_legal_dialect("llvm")
    |> Plan.add_illegal_dialect("ex")
    |> Plan.add_conversion(&convert_type/1, version: "1.0")
    |> Plan.add_conversion_pattern("ex.lit", &convert_lit/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.add", &convert_add/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.sub", &convert_sub/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mul", &convert_mul/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.div", &convert_div/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.rem", &convert_rem/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cmp", &convert_cmp/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.if", &convert_if/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.yield", &convert_yield/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.call", &convert_call/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.return", &convert_return/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.var", &convert_var/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.func", &convert_func/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.box", &convert_box/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.to_word", &convert_to_word/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.unbox", &convert_to_word/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.self", &convert_self/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.send", &convert_send/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.receive", &convert_receive/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mailbox_clear", &convert_mailbox_clear/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.spawn", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.runtime_create", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.runtime_enter", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.runtime_leave", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.runtime_destroy", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.result_create", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.result_destroy", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.result_root_kind", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.result_root_word", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.result_exception_kind", &convert_term_read/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.result_exception_reason", &convert_term_read/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.result_term_kind", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.result_term_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.result_term_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.term_export", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.term_import", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.exported_clone", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.exported_destroy", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.exported_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.exported_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.term_handle_export", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.term_handle_destroy", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.process_table_reset", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cont_save", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.receive_cont_save", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cont_pending", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cont_active", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cont_clear", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cont_load_arg", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cont_load_acc", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.cont_load_cursor", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.schedule_next", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.worker_run", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.process_wait", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mailbox_len", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mailbox_peek", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mailbox_remove", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.nil_word", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.monotonic_time", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.receive_start", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.receive_start_set", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.native_time", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.unique_integer", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.current_entry", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.process_done", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.process_exit", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.process_exit_reason", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.process_trap_exit", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.link", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.unlink", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.exit", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.monitor", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.demonitor", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.processes_runnable", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.process_result", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.to_int", &convert_to_int/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.reduction_tick", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.clock_init", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.yield_mark", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.try", &convert_try/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.throw", &convert_throw/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.raise", &convert_raise/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.catch_value", &convert_catch_value/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.make_fun", &convert_make_fun/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.make_fun_with_arity", &convert_make_fun_with_arity/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.fun_arity", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.apply", &convert_apply/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.tuple", &convert_term_tuple/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list", &convert_term_list/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list_cons", &convert_term_list_cons/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.map", &convert_term_map/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.map_put", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.map_fetch", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mapset_from_list", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mapset_member", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.mapset_put", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.file_read", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.file_read_lines", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary", &convert_term_binary/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_from_list", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.iodata_to_binary", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.float_lit", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.string_to_float", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_integer", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_float", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_atom", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_binary", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_list", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_tuple", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.is_map", &convert_term_predicate/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.tuple_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.tuple_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.map_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.enumerable_count", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.enumerable_to_list", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.enumerable_into_map", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.enumerable_intersperse", &convert_term_read/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.enumerable_to_list_range", &convert_term_read/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.enumerable_reduce", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.enumerable_reduce_c", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.enumerable_reduce_range", &convert_term_read/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.enumerable_reduce_fun", &convert_term_read/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.enumerable_map_fun", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.enumerable_map_term_fun", &convert_term_read/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.enumerable_flat_map_term_fun", &convert_term_read/3,
      version: "1.0"
    )
    |> Plan.add_conversion_pattern("ex.stream_filter", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.stream_take", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.stream_drop", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.func_addr", &convert_func_addr/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list_head", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list_tail", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.list_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.term_eq", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.term_eq_loose", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_slice", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_utf8_get", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_utf8_width", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_utf8_length", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.string_printable", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_quote", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_encode16", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.binary_decode16", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.int_to_string", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.int_to_string_base", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.int_to_hex", &convert_term_read/3, version: "1.0")
    |> Plan.add_conversion_pattern("ex.string_to_int", &convert_term_read/3, version: "1.0")
  end

  # Declaration-first manifest of the Zig term runtime ABI: batata's
  # `native/term_runtime.zig` exports exactly these C symbols.
  @term_intrinsics %{
    list_cons: "ex.term.list_cons",
    map_put: "ex.term.map_put",
    map_fetch: "ex.term.map_fetch",
    self: "ex.term.self",
    send: "ex.term.send",
    receive: "ex.term.receive",
    mailbox_clear: "ex.term.mailbox_clear",
    spawn: "ex.term.spawn",
    runtime_create: "ex.term.runtime_create",
    runtime_enter: "ex.term.runtime_enter",
    runtime_leave: "ex.term.runtime_leave",
    runtime_destroy: "ex.term.runtime_destroy",
    result_create: "ex.term.result_create",
    result_destroy: "ex.term.result_destroy",
    result_root_kind: "ex.term.result_root_kind",
    result_root_word: "ex.term.result_root_word",
    result_exception_kind: "ex.term.result_exception_kind",
    result_exception_reason: "ex.term.result_exception_reason",
    result_term_kind: "ex.term.result_term_kind",
    result_term_length: "ex.term.result_term_length",
    result_term_get: "ex.term.result_term_get",
    term_export: "ex.term.export",
    term_import: "ex.term.import",
    exported_clone: "ex.term.exported_clone",
    exported_destroy: "ex.term.exported_destroy",
    exported_length: "ex.term.exported_length",
    exported_get: "ex.term.exported_get",
    term_handle_export: "ex.term.handle_export",
    term_handle_destroy: "ex.term.handle_destroy",
    process_table_reset: "ex.term.process_table_reset",
    cont_save: "ex.term.cont_save",
    receive_cont_save: "ex.term.receive_cont_save",
    cont_pending: "ex.term.cont_pending",
    cont_active: "ex.term.cont_active",
    cont_clear: "ex.term.cont_clear",
    cont_load_arg: "ex.term.cont_load_arg",
    cont_load_acc: "ex.term.cont_load_acc",
    cont_load_cursor: "ex.term.cont_load_cursor",
    schedule_next: "ex.term.schedule_next",
    mailbox_len: "ex.term.mailbox_len",
    mailbox_peek: "ex.term.mailbox_peek",
    mailbox_remove: "ex.term.mailbox_remove",
    nil_word: "ex.term.nil",
    monotonic_time: "ex.term.monotonic_time",
    receive_start: "ex.term.receive_start",
    receive_start_set: "ex.term.receive_start_set",
    native_time: "ex.term.native_time",
    unique_integer: "ex.term.unique_integer",
    current_entry: "ex.term.current_entry",
    process_done: "ex.term.process_done",
    process_exit: "ex.term.process_exit",
    process_exit_reason: "ex.term.process_exit_reason",
    process_trap_exit: "ex.term.process_trap_exit",
    link: "ex.term.link",
    unlink: "ex.term.unlink",
    exit: "ex.term.exit",
    monitor: "ex.term.monitor",
    demonitor: "ex.term.demonitor",
    processes_runnable: "ex.term.processes_runnable",
    process_result: "ex.term.process_result",
    worker_run: "ex.term.worker_run",
    process_wait: "ex.term.process_wait",
    to_int: "ex.term.to_int",
    reduction_tick: "ex.term.clock_tick",
    clock_init: "ex.term.clock_init",
    yield_mark: "ex.term.yield_mark",
    jmp_buf_size: "ex.term.jmp_buf_size",
    setjmp_addr: "ex.term.setjmp_addr",
    try_push: "ex.term.try_push",
    try_pop: "ex.term.try_pop",
    throw: "ex.term.throw",
    raise: "ex.term.raise",
    catch_value: "ex.term.catch_value",
    make_fun: "ex.term.make_fun",
    make_fun_with_arity: "ex.term.make_fun_with_arity",
    fun_arity: "ex.term.fun_arity",
    fun_idx: "ex.term.fun_idx",
    fun_env: "ex.term.fun_env",
    tuple_from_list: "ex.term.tuple_from_list",
    map_from_list: "ex.term.map_from_list",
    mapset_from_list: "ex.term.mapset_from_list",
    mapset_member: "ex.term.mapset_member",
    mapset_put: "ex.term.mapset_put",
    file_read: "ex.term.file_read",
    file_read_lines: "ex.term.file_read_lines",
    binary_from_list: "ex.term.binary_from_list",
    iodata_to_binary: "ex.term.iodata_to_binary",
    float_lit: "ex.term.float_lit",
    string_to_float: "ex.term.string_to_float",
    is_integer: "ex.term.is_integer",
    is_float: "ex.term.is_float",
    is_atom: "ex.term.is_atom",
    is_binary: "ex.term.is_binary",
    is_list: "ex.term.is_list",
    is_tuple: "ex.term.is_tuple",
    is_map: "ex.term.is_map",
    tuple_get: "ex.term.tuple_get",
    tuple_length: "ex.term.tuple_length",
    map_length: "ex.term.map_length",
    enumerable_count: "ex.term.enumerable_count",
    enumerable_to_list: "ex.term.enumerable_to_list",
    enumerable_into_map: "ex.term.enumerable_into_map",
    enumerable_intersperse: "ex.term.enumerable_intersperse",
    enumerable_to_list_range: "ex.term.enumerable_to_list_range",
    enumerable_reduce: "ex.term.enumerable_reduce",
    enumerable_reduce_c: "ex.term.enumerable_reduce_c",
    enumerable_reduce_range: "ex.term.enumerable_reduce_range",
    enumerable_reduce_fun: "ex.term.enumerable_reduce_fun",
    enumerable_map_fun: "ex.term.enumerable_map_fun",
    enumerable_map_term_fun: "ex.term.enumerable_map_term_fun",
    enumerable_flat_map_term_fun: "ex.term.enumerable_flat_map_term_fun",
    stream_filter: "ex.term.stream_filter",
    stream_take: "ex.term.stream_take",
    stream_drop: "ex.term.stream_drop",
    list_head: "ex.term.list_head",
    list_tail: "ex.term.list_tail",
    list_get: "ex.term.list_get",
    list_length: "ex.term.list_length",
    term_eq: "ex.term.eq",
    term_eq_loose: "ex.term.eq_loose",
    binary_length: "ex.term.binary_length",
    binary_get: "ex.term.binary_get",
    binary_slice: "ex.term.binary_slice",
    binary_utf8_get: "ex.term.binary_utf8_get",
    binary_utf8_width: "ex.term.binary_utf8_width",
    binary_utf8_length: "ex.term.binary_utf8_length",
    string_printable: "ex.term.string_printable",
    binary_quote: "ex.term.binary_quote",
    binary_encode16: "ex.term.binary_encode16",
    binary_decode16: "ex.term.binary_decode16",
    int_to_string: "ex.term.int_to_string",
    int_to_string_base: "ex.term.int_to_string_base",
    int_to_hex: "ex.term.int_to_hex",
    string_to_int: "ex.term.string_to_int"
  }

  @doc """
  Returns the term runtime intrinsic symbols emitted by this conversion plan.

  This is the declaration-first anchor for host-boundary manifests (e.g.
  `Beaver.Wasm.TermABI`): every symbol here must be provided by the term
  runtime (batata's Zig on native, a wasm host import on wasm).
  """
  @spec term_intrinsic_symbols() :: [String.t()]
  def term_intrinsic_symbols do
    @term_intrinsics |> Map.values() |> Enum.sort()
  end

  @term_types ~w(!ex.term !ex.bound !ex.unbound)

  @doc """
  Converts an `ex` term type to its scalar word representation.
  """
  @spec convert_type(MLIR.Type.t()) :: MLIR.Type.t()
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

  defp convert_div(operation, [left, right], rewriter) do
    convert_binary("arith.divsi", operation, [left, right], rewriter)
  end

  defp convert_rem(operation, [left, right], rewriter) do
    convert_binary("arith.remsi", operation, [left, right], rewriter)
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

  defp convert_term_list_cons(operation, [head, tail], rewriter) do
    base = insertion_point(operation, rewriter)

    result =
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.list_cons, [head, tail])

    replace_with(rewriter, operation, result)
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

    replace_with(
      rewriter,
      operation,
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.self, [])
    )
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

  # `ex.func_addr` resolves a function symbol to its address (an i64 word):
  # func.constant yields the function value, which func-to-llvm lowers to a
  # pointer. Used to hand a compiled reducer to the runtime's
  # arbitrary-closure enumerable reduce.
  defp convert_func_addr(operation, [], rewriter) do
    context = MLIR.context(operation)
    base = insertion_point(operation, rewriter)
    location = MLIR.Operation.location(operation)
    {:ok, sym_name_attr} = operation |> MLIR.Operation.fetch(:sym_name)
    name = sym_name_attr |> MLIR.CAPI.mlirStringAttrGetValue() |> MLIR.to_string()
    [result] = operation |> Walker.results() |> Enum.to_list()
    result_type = result |> MLIR.Value.type()

    func_const =
      %Changeset{name: "func.constant", context: context, location: location}
      |> Changeset.add_argument(value: MLIR.Attribute.flat_symbol_ref(name, ctx: context))
      |> Changeset.add_result(result_type)
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, func_const)
    replace_with(rewriter, operation, MLIR.Operation.result(func_const, 0))
  end

  # `ex.try` lowers to a setjmp/longjmp pair: allocate a jmp_buf on the stack,
  # push it on the runtime's buffer stack, and call libc `setjmp`. The normal
  # path runs the body region; a `throw` longjmps back and the catch region
  # matches the thrown value.
  defp convert_try(operation, [], rewriter) do
    context = MLIR.context(operation)
    base = insertion_point(operation, rewriter)
    location = MLIR.Operation.location(operation)
    [body_region, catch_region] = operation |> Walker.regions() |> Enum.to_list()

    size = emit_runtime_call(operation, rewriter, base, @term_intrinsics.jmp_buf_size, [])

    alloca =
      %Changeset{name: "llvm.alloca", context: context, location: location}
      |> Changeset.add_argument([size])
      |> Changeset.add_argument(
        elem_type: MLIR.Attribute.type(MLIR.Type.integer(8, ctx: context)),
        alignment: MLIR.Attribute.integer(MLIR.Type.i64(), 8)
      )
      |> Changeset.add_result(MLIR.Type.llvm_pointer(ctx: context))
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, alloca)
    buf = MLIR.Operation.result(alloca, 0)

    push = emit_runtime_call(operation, rewriter, base, @term_intrinsics.try_push, [buf])
    _ = push

    # libc's setjmp is resolved indirectly (the ORC linker does not resolve
    # libc symbols on its own): the runtime exposes its address.
    setjmp_addr = emit_runtime_call(operation, rewriter, base, @term_intrinsics.setjmp_addr, [])

    fn_ptr =
      %Changeset{name: "llvm.inttoptr", context: context, location: location}
      |> Changeset.add_argument([setjmp_addr])
      |> Changeset.add_result(MLIR.Type.llvm_pointer(ctx: context))
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, fn_ptr)

    setjmp =
      %Changeset{name: "llvm.call", context: context, location: location}
      |> Changeset.add_argument([MLIR.Operation.result(fn_ptr, 0), buf])
      |> Changeset.add_argument(
        operandSegmentSizes: MLIR.Attribute.dense_array([2, 0], Beaver.Native.I32, ctx: context),
        op_bundle_sizes: MLIR.Attribute.dense_array([], Beaver.Native.I32, ctx: context)
      )
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, setjmp)
    saved = MLIR.Operation.result(setjmp, 0)

    zero = emit_constant(0, operation, base) |> MLIR.Operation.result(0)

    cmp =
      %Changeset{name: "arith.cmpi", context: context, location: location}
      |> Changeset.add_argument([saved, zero])
      |> Changeset.add_argument(predicate: cmp_i_predicate_attr(0))
      |> Changeset.add_result(MLIR.Type.i1())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, cmp)

    scf_if =
      %Changeset{name: "scf.if", context: context, location: location}
      |> Changeset.add_argument(MLIR.Operation.result(cmp, 0))
      |> Changeset.add_argument(MLIR.CAPI.mlirRegionCreate())
      |> Changeset.add_argument(MLIR.CAPI.mlirRegionCreate())
      |> Changeset.add_result([MLIR.Type.i64()])
      |> MLIR.Operation.create()

    [new_then, new_else] = scf_if |> Walker.regions() |> Enum.to_list()
    MLIR.CAPI.mlirRegionTakeBody(new_then, body_region)
    MLIR.CAPI.mlirRegionTakeBody(new_else, catch_region)

    prepend_try_pop(operation, rewriter, new_else)
    append_try_pop(operation, rewriter, new_then)

    MLIR.RewriterBase.set_insertion_point_before(base, operation)
    MLIR.RewriterBase.insert(base, scf_if)

    MLIR.ConversionPatternRewriter.replace_op(
      rewriter,
      operation,
      scf_if |> Walker.results() |> Enum.to_list()
    )

    :ok
  end

  # The catch path pops the try buffer first: the longjmp returns with the
  # buffer still pushed.
  defp prepend_try_pop(operation, rewriter, region) do
    ensure_intrinsic_declaration(operation, rewriter, @term_intrinsics.try_pop)
    [block] = region |> Walker.blocks() |> Enum.to_list()
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    MLIR.RewriterBase.set_insertion_point_to_start(base, block)
    emit_try_pop_call(operation, rewriter)
    :ok
  end

  # The normal path pops the try buffer before leaving the body.
  defp append_try_pop(operation, rewriter, region) do
    ensure_intrinsic_declaration(operation, rewriter, @term_intrinsics.try_pop)
    [block] = region |> Walker.blocks() |> Enum.to_list()
    terminator = MLIR.CAPI.mlirBlockGetTerminator(block)
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    MLIR.RewriterBase.set_insertion_point_before(base, terminator)
    emit_try_pop_call(operation, rewriter)
    :ok
  end

  # Builds a func.call to the try_pop intrinsic at the current insertion
  # point (unlike `emit_runtime_call`, this does not move the insertion
  # point, so it can be placed inside a region).
  defp emit_try_pop_call(operation, rewriter) do
    base = MLIR.ConversionPatternRewriter.as_base(rewriter)
    symbol = @term_intrinsics.try_pop

    call =
      %Changeset{
        name: "func.call",
        context: MLIR.context(operation),
        location: MLIR.Operation.location(operation)
      }
      |> Changeset.add_argument([])
      |> Changeset.add_argument(
        callee: MLIR.Attribute.flat_symbol_ref(symbol, ctx: MLIR.context(operation))
      )
      |> Changeset.add_result(MLIR.Type.i64())
      |> MLIR.Operation.create()

    MLIR.RewriterBase.insert(base, call)
    MLIR.Operation.result(call, 0)
  end

  defp convert_throw(operation, [operand], rewriter) do
    base = insertion_point(operation, rewriter)

    replace_with(
      rewriter,
      operation,
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.throw, [operand])
    )
  end

  defp convert_raise(operation, [reason, kind], rewriter) do
    base = insertion_point(operation, rewriter)

    replace_with(
      rewriter,
      operation,
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.raise, [reason, kind])
    )
  end

  defp convert_catch_value(operation, [], rewriter) do
    base = insertion_point(operation, rewriter)

    replace_with(
      rewriter,
      operation,
      emit_runtime_call(operation, rewriter, base, @term_intrinsics.catch_value, [])
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

  # Additive arity-carrying constructor. The legacy conversion above remains
  # byte-for-byte compatible with runtimes that only export `make_fun/6`.
  defp convert_make_fun_with_arity(operation, operands, rewriter) do
    base = insertion_point(operation, rewriter)

    fn_idx = operation |> required_attribute("fn_idx") |> MLIR.CAPI.mlirIntegerAttrGetValueInt()
    arity = operation |> required_attribute("arity") |> MLIR.CAPI.mlirIntegerAttrGetValueInt()
    env_len = operation |> required_attribute("env_len") |> MLIR.CAPI.mlirIntegerAttrGetValueInt()

    fn_idx = Beaver.Native.to_term(fn_idx)
    arity = Beaver.Native.to_term(arity)
    env_len = Beaver.Native.to_term(env_len)

    unless length(operands) == env_len do
      raise ArgumentError,
            "ex.make_fun_with_arity env_len #{env_len} does not match #{length(operands)} operands"
    end

    unless arity in 0..4 do
      raise ArgumentError, "ex.make_fun_with_arity supports arities 0..4, got #{arity}"
    end

    constants =
      Enum.map([fn_idx, arity, env_len], fn value ->
        emit_constant(value, operation, base) |> MLIR.Operation.result(0)
      end)

    pad =
      List.duplicate(emit_constant(0, operation, base) |> MLIR.Operation.result(0), 4 - env_len)

    closure =
      emit_runtime_call(
        operation,
        rewriter,
        base,
        @term_intrinsics.make_fun_with_arity,
        constants ++ operands ++ pad
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
  defp predicate_intrinsic("ex.is_float"), do: @term_intrinsics.is_float
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
  defp read_intrinsic("ex.fun_arity"), do: @term_intrinsics.fun_arity
  defp read_intrinsic("ex.tuple_length"), do: @term_intrinsics.tuple_length
  defp read_intrinsic("ex.map_length"), do: @term_intrinsics.map_length
  defp read_intrinsic("ex.float_lit"), do: @term_intrinsics.float_lit
  defp read_intrinsic("ex.string_to_float"), do: @term_intrinsics.string_to_float
  defp read_intrinsic("ex.map_put"), do: @term_intrinsics.map_put
  defp read_intrinsic("ex.map_fetch"), do: @term_intrinsics.map_fetch
  defp read_intrinsic("ex.mapset_from_list"), do: @term_intrinsics.mapset_from_list
  defp read_intrinsic("ex.mapset_member"), do: @term_intrinsics.mapset_member
  defp read_intrinsic("ex.mapset_put"), do: @term_intrinsics.mapset_put
  defp read_intrinsic("ex.file_read"), do: @term_intrinsics.file_read
  defp read_intrinsic("ex.file_read_lines"), do: @term_intrinsics.file_read_lines
  defp read_intrinsic("ex.enumerable_count"), do: @term_intrinsics.enumerable_count
  defp read_intrinsic("ex.enumerable_to_list"), do: @term_intrinsics.enumerable_to_list
  defp read_intrinsic("ex.enumerable_into_map"), do: @term_intrinsics.enumerable_into_map
  defp read_intrinsic("ex.enumerable_intersperse"), do: @term_intrinsics.enumerable_intersperse

  defp read_intrinsic("ex.enumerable_to_list_range"),
    do: @term_intrinsics.enumerable_to_list_range

  defp read_intrinsic("ex.enumerable_reduce"), do: @term_intrinsics.enumerable_reduce
  defp read_intrinsic("ex.enumerable_reduce_c"), do: @term_intrinsics.enumerable_reduce_c

  defp read_intrinsic("ex.enumerable_reduce_range"),
    do: @term_intrinsics.enumerable_reduce_range

  defp read_intrinsic("ex.enumerable_reduce_fun"), do: @term_intrinsics.enumerable_reduce_fun
  defp read_intrinsic("ex.enumerable_map_fun"), do: @term_intrinsics.enumerable_map_fun

  defp read_intrinsic("ex.enumerable_map_term_fun"),
    do: @term_intrinsics.enumerable_map_term_fun

  defp read_intrinsic("ex.enumerable_flat_map_term_fun"),
    do: @term_intrinsics.enumerable_flat_map_term_fun

  defp read_intrinsic("ex.stream_filter"), do: @term_intrinsics.stream_filter
  defp read_intrinsic("ex.stream_take"), do: @term_intrinsics.stream_take
  defp read_intrinsic("ex.stream_drop"), do: @term_intrinsics.stream_drop

  defp read_intrinsic("ex.list_head"), do: @term_intrinsics.list_head
  defp read_intrinsic("ex.list_tail"), do: @term_intrinsics.list_tail
  defp read_intrinsic("ex.list_get"), do: @term_intrinsics.list_get
  defp read_intrinsic("ex.list_length"), do: @term_intrinsics.list_length
  defp read_intrinsic("ex.term_eq"), do: @term_intrinsics.term_eq
  defp read_intrinsic("ex.term_eq_loose"), do: @term_intrinsics.term_eq_loose
  defp read_intrinsic("ex.reduction_tick"), do: @term_intrinsics.reduction_tick
  defp read_intrinsic("ex.clock_init"), do: @term_intrinsics.clock_init
  defp read_intrinsic("ex.yield_mark"), do: @term_intrinsics.yield_mark
  defp read_intrinsic("ex.spawn"), do: @term_intrinsics.spawn
  defp read_intrinsic("ex.runtime_create"), do: @term_intrinsics.runtime_create
  defp read_intrinsic("ex.runtime_enter"), do: @term_intrinsics.runtime_enter
  defp read_intrinsic("ex.runtime_leave"), do: @term_intrinsics.runtime_leave
  defp read_intrinsic("ex.runtime_destroy"), do: @term_intrinsics.runtime_destroy
  defp read_intrinsic("ex.result_create"), do: @term_intrinsics.result_create
  defp read_intrinsic("ex.result_destroy"), do: @term_intrinsics.result_destroy
  defp read_intrinsic("ex.result_root_kind"), do: @term_intrinsics.result_root_kind
  defp read_intrinsic("ex.result_root_word"), do: @term_intrinsics.result_root_word
  defp read_intrinsic("ex.result_exception_kind"), do: @term_intrinsics.result_exception_kind
  defp read_intrinsic("ex.result_exception_reason"), do: @term_intrinsics.result_exception_reason
  defp read_intrinsic("ex.result_term_kind"), do: @term_intrinsics.result_term_kind
  defp read_intrinsic("ex.result_term_length"), do: @term_intrinsics.result_term_length
  defp read_intrinsic("ex.result_term_get"), do: @term_intrinsics.result_term_get
  defp read_intrinsic("ex.term_export"), do: @term_intrinsics.term_export
  defp read_intrinsic("ex.term_import"), do: @term_intrinsics.term_import
  defp read_intrinsic("ex.exported_clone"), do: @term_intrinsics.exported_clone
  defp read_intrinsic("ex.exported_destroy"), do: @term_intrinsics.exported_destroy
  defp read_intrinsic("ex.exported_length"), do: @term_intrinsics.exported_length
  defp read_intrinsic("ex.exported_get"), do: @term_intrinsics.exported_get
  defp read_intrinsic("ex.term_handle_export"), do: @term_intrinsics.term_handle_export
  defp read_intrinsic("ex.term_handle_destroy"), do: @term_intrinsics.term_handle_destroy
  defp read_intrinsic("ex.process_table_reset"), do: @term_intrinsics.process_table_reset
  defp read_intrinsic("ex.cont_save"), do: @term_intrinsics.cont_save
  defp read_intrinsic("ex.receive_cont_save"), do: @term_intrinsics.receive_cont_save
  defp read_intrinsic("ex.cont_pending"), do: @term_intrinsics.cont_pending
  defp read_intrinsic("ex.cont_active"), do: @term_intrinsics.cont_active
  defp read_intrinsic("ex.cont_clear"), do: @term_intrinsics.cont_clear
  defp read_intrinsic("ex.cont_load_arg"), do: @term_intrinsics.cont_load_arg
  defp read_intrinsic("ex.cont_load_acc"), do: @term_intrinsics.cont_load_acc
  defp read_intrinsic("ex.cont_load_cursor"), do: @term_intrinsics.cont_load_cursor
  defp read_intrinsic("ex.schedule_next"), do: @term_intrinsics.schedule_next
  defp read_intrinsic("ex.worker_run"), do: @term_intrinsics.worker_run
  defp read_intrinsic("ex.process_wait"), do: @term_intrinsics.process_wait
  defp read_intrinsic("ex.mailbox_len"), do: @term_intrinsics.mailbox_len
  defp read_intrinsic("ex.mailbox_peek"), do: @term_intrinsics.mailbox_peek
  defp read_intrinsic("ex.mailbox_remove"), do: @term_intrinsics.mailbox_remove
  defp read_intrinsic("ex.nil_word"), do: @term_intrinsics.nil_word
  defp read_intrinsic("ex.monotonic_time"), do: @term_intrinsics.monotonic_time
  defp read_intrinsic("ex.receive_start"), do: @term_intrinsics.receive_start
  defp read_intrinsic("ex.receive_start_set"), do: @term_intrinsics.receive_start_set
  defp read_intrinsic("ex.native_time"), do: @term_intrinsics.native_time
  defp read_intrinsic("ex.unique_integer"), do: @term_intrinsics.unique_integer
  defp read_intrinsic("ex.current_entry"), do: @term_intrinsics.current_entry
  defp read_intrinsic("ex.process_done"), do: @term_intrinsics.process_done
  defp read_intrinsic("ex.process_exit"), do: @term_intrinsics.process_exit
  defp read_intrinsic("ex.process_exit_reason"), do: @term_intrinsics.process_exit_reason
  defp read_intrinsic("ex.process_trap_exit"), do: @term_intrinsics.process_trap_exit
  defp read_intrinsic("ex.link"), do: @term_intrinsics.link
  defp read_intrinsic("ex.unlink"), do: @term_intrinsics.unlink
  defp read_intrinsic("ex.exit"), do: @term_intrinsics.exit
  defp read_intrinsic("ex.monitor"), do: @term_intrinsics.monitor
  defp read_intrinsic("ex.demonitor"), do: @term_intrinsics.demonitor
  defp read_intrinsic("ex.processes_runnable"), do: @term_intrinsics.processes_runnable
  defp read_intrinsic("ex.process_result"), do: @term_intrinsics.process_result
  defp read_intrinsic("ex.binary_length"), do: @term_intrinsics.binary_length
  defp read_intrinsic("ex.binary_from_list"), do: @term_intrinsics.binary_from_list
  defp read_intrinsic("ex.iodata_to_binary"), do: @term_intrinsics.iodata_to_binary
  defp read_intrinsic("ex.binary_get"), do: @term_intrinsics.binary_get
  defp read_intrinsic("ex.binary_slice"), do: @term_intrinsics.binary_slice
  defp read_intrinsic("ex.binary_utf8_get"), do: @term_intrinsics.binary_utf8_get
  defp read_intrinsic("ex.binary_utf8_width"), do: @term_intrinsics.binary_utf8_width
  defp read_intrinsic("ex.binary_utf8_length"), do: @term_intrinsics.binary_utf8_length
  defp read_intrinsic("ex.string_printable"), do: @term_intrinsics.string_printable
  defp read_intrinsic("ex.binary_quote"), do: @term_intrinsics.binary_quote
  defp read_intrinsic("ex.binary_encode16"), do: @term_intrinsics.binary_encode16
  defp read_intrinsic("ex.binary_decode16"), do: @term_intrinsics.binary_decode16
  defp read_intrinsic("ex.int_to_string"), do: @term_intrinsics.int_to_string
  defp read_intrinsic("ex.int_to_string_base"), do: @term_intrinsics.int_to_string_base
  defp read_intrinsic("ex.int_to_hex"), do: @term_intrinsics.int_to_hex
  defp read_intrinsic("ex.string_to_int"), do: @term_intrinsics.string_to_int

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
        |> Changeset.add_argument(
          function_type: intrinsic_function_type(symbol, MLIR.context(operation))
        )
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

  defp intrinsic_function_type("ex.term.list_cons", _ctx) do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.map_put", _ctx) do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.self", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.send", _ctx) do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.receive", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.yield_mark", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.mailbox_clear", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.spawn", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.runtime_create", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.runtime_enter", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.runtime_leave", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.runtime_destroy", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.result_create", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 2), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(symbol, _ctx)
       when symbol in [
              "ex.term.result_destroy",
              "ex.term.result_root_kind",
              "ex.term.result_root_word",
              "ex.term.result_exception_kind",
              "ex.term.result_exception_reason"
            ] do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(symbol, _ctx)
       when symbol in ["ex.term.result_term_kind", "ex.term.result_term_length"] do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 2), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.result_term_get", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 3), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(symbol, _ctx)
       when symbol in ["ex.term.export", "ex.term.import", "ex.term.exported_get"] do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 2), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(symbol, _ctx)
       when symbol in [
              "ex.term.exported_clone",
              "ex.term.exported_destroy",
              "ex.term.exported_length",
              "ex.term.handle_export",
              "ex.term.handle_destroy"
            ] do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.process_table_reset", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.cont_save", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 3), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.receive_cont_save", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 3), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.cont_pending", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.cont_active", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.cont_clear", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.cont_load_arg", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.cont_load_acc", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.cont_load_cursor", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.schedule_next", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.mailbox_len", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.mailbox_peek", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.mailbox_remove", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.nil", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.monotonic_time", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.receive_start", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.receive_start_set", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.native_time", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.unique_integer", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.current_entry", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.process_done", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(symbol, _ctx)
       when symbol in ["ex.term.process_exit", "ex.term.process_exit_reason"] do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.process_trap_exit", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.link", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 3), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.unlink", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(symbol, _ctx)
       when symbol in ["ex.term.exit", "ex.term.monitor"] do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 4), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.demonitor", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.processes_runnable", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.process_result", _ctx) do
    MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.jmp_buf_size", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.setjmp_addr", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.try_push", ctx) do
    MLIR.Type.function([MLIR.Type.llvm_pointer(ctx: ctx)], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.try_pop", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.catch_value", _ctx) do
    MLIR.Type.function([], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.raise", _ctx) do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.make_fun", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 6), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.make_fun_with_arity", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 7), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.enumerable_reduce", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 3), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.enumerable_reduce_c", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 4), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.enumerable_reduce_range", _ctx) do
    MLIR.Type.function(List.duplicate(MLIR.Type.i64(), 4), [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.enumerable_reduce_fun", _ctx) do
    MLIR.Type.function(
      [
        MLIR.Type.i64(),
        MLIR.Type.i64(),
        MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
      ],
      [MLIR.Type.i64()]
    )
  end

  defp intrinsic_function_type("ex.term.worker_run", _ctx) do
    MLIR.Type.function(
      [MLIR.Type.i64(), MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])],
      [MLIR.Type.i64()]
    )
  end

  defp intrinsic_function_type("ex.term.enumerable_map_fun", _ctx) do
    MLIR.Type.function(
      [
        MLIR.Type.i64(),
        MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
      ],
      [MLIR.Type.i64()]
    )
  end

  defp intrinsic_function_type("ex.term.enumerable_map_term_fun", _ctx) do
    MLIR.Type.function(
      [
        MLIR.Type.i64(),
        MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
      ],
      [MLIR.Type.i64()]
    )
  end

  defp intrinsic_function_type("ex.term.enumerable_flat_map_term_fun", _ctx) do
    MLIR.Type.function(
      [
        MLIR.Type.i64(),
        MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
      ],
      [MLIR.Type.i64()]
    )
  end

  defp intrinsic_function_type("ex.term.stream_filter", _ctx) do
    MLIR.Type.function(
      [
        MLIR.Type.i64(),
        MLIR.Type.function([MLIR.Type.i64()], [MLIR.Type.i64()])
      ],
      [MLIR.Type.i64()]
    )
  end

  defp intrinsic_function_type("ex.term.enumerable_to_list_range", _ctx) do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type("ex.term.fun_env", _ctx) do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(symbol, _ctx)
       when symbol in [
              "ex.term.tuple_get",
              "ex.term.list_get",
              "ex.term.eq",
              "ex.term.eq_loose",
              "ex.term.binary_get",
              "ex.term.binary_slice",
              "ex.term.binary_utf8_get",
              "ex.term.binary_utf8_width",
              "ex.term.int_to_string_base",
              "ex.term.enumerable_into_map",
              "ex.term.enumerable_intersperse",
              "ex.term.map_fetch",
              "ex.term.mapset_member",
              "ex.term.mapset_put",
              "ex.term.stream_take",
              "ex.term.stream_drop"
            ] do
    MLIR.Type.function([MLIR.Type.i64(), MLIR.Type.i64()], [MLIR.Type.i64()])
  end

  defp intrinsic_function_type(_symbol, _ctx) do
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
