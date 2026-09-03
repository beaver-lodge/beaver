defmodule Beaver.MLIR.Dialect.Ex do
  @moduledoc """
  The `ex` dialect: a scalar subset of Elixir AST defined in Slang.

  M0 implements the scalar subset of the planned Elixir dialect:

    * `ex.lit` - integer literals with an integer value attribute
    * `ex.var` / `ex.bind` - named variables and their bindings
    * `ex.add` - typed scalar addition
    * `ex.call` - local calls with callee and arity attributes
    * `ex.func` / `ex.return` - functions with an isolated body region
      terminated by `ex.return`

  The dialect is defined entirely with `Beaver.Slang` and replaces the native
  TableGen `elixir` dialect prototype. Load it into a context with
  `Beaver.Slang.load/2`.

  The `ex.if` op is declared as `defop if(...)`; the parentheses are required
  by the Slang `defop` DSL and are not an Elixir `if/2` call.
  """

  @external_resource __ENV__.file
  @schema_digest "sha256:" <>
                   Base.encode16(:crypto.hash(:sha256, File.read!(__ENV__.file)), case: :lower)

  use Beaver.Slang, name: "ex"

  @doc "Returns the content identity of this Ex dialect schema definition."
  @spec schema_digest() :: String.t()
  def schema_digest, do: @schema_digest

  deftype term()
  deftype bound()
  deftype unbound()

  defconstraint integer_value do
    base("#builtin.integer")
  end

  defconstraint callee_value do
    base("#builtin.string")
  end

  defconstraint cmp_predicate_value do
    base("#builtin.string")
  end

  # Named constraints keep the generated creator functions compact. Inlining
  # these expressions repeats their MLIR builder AST in every operation.
  defconstraint any_value do
    any()
  end

  defconstraint integer_like do
    all_of([base("!builtin.integer"), any()])
  end

  defconstraint term_like do
    any_of([base(term()), base(bound()), base(unbound())])
  end

  defop lit(),
    results: [result: ^integer_like],
    attributes: [value: ^integer_value]

  defop var(),
    results: [result: base(unbound())],
    attributes: [name: base("#builtin.string")]

  defop bind(variable = base(unbound()), value = ^any_value),
    results: [result: base(bound())]

  defop add(
          left = ^integer_like,
          right = ^integer_like
        ),
        results: [result: ^integer_like]

  defop sub(
          left = ^integer_like,
          right = ^integer_like
        ),
        results: [result: ^integer_like]

  defop mul(
          left = ^integer_like,
          right = ^integer_like
        ),
        results: [result: ^integer_like]

  defop div(
          left = ^integer_like,
          right = ^integer_like
        ),
        results: [result: ^integer_like]

  defop rem(
          left = ^integer_like,
          right = ^integer_like
        ),
        results: [result: ^integer_like]

  # Each argument is its own optional slot so heterogeneous argument types
  # (e.g. a term plus a scalar accumulator) verify: IRDL variadic groups are
  # homogeneous, which would reject mixed-typed calls. Eight slots cover the
  # closure ABI: four captured values plus four application arguments.
  defop call(
          arg0 = optional(^any_value),
          arg1 = optional(^any_value),
          arg2 = optional(^any_value),
          arg3 = optional(^any_value),
          arg4 = optional(^any_value),
          arg5 = optional(^any_value),
          arg6 = optional(^any_value),
          arg7 = optional(^any_value)
        ),
        results: [result: ^any_value],
        attributes: [callee: ^callee_value, arity: ^integer_value]

  defop cmp(
          left = ^integer_like,
          right = ^integer_like
        ),
        results: [result: ^integer_like],
        attributes: [predicate: ^cmp_predicate_value]

  # credo:disable-for-next-line Credo.Check.Readability.ParenthesesInCondition
  defop if(cond = ^integer_like),
    results: [result: variadic(^any_value)],
    regions: [:any, :any]

  defop case(scrutinee = ^any_value),
    results: [result: variadic(^any_value)],
    regions: [:any]

  defop clause(guard = optional(^any_value)),
    attributes: [patterns: ^any_value]

  defop box(value = ^any_value),
    results: [result: ^term_like]

  # Lifts an already-tagged word (e.g. a function argument) into the term
  # type without tagging: the conversion is a pure passthrough.
  defop to_word(value = ^any_value),
    results: [result: ^term_like]

  # Drops the term type annotation without untagging: the conversion is a
  # pure passthrough, so a term word can cross control-flow regions (whose
  # types must be legal after conversion) as a scalar i64.
  defop unbox(word = ^term_like),
    results: [result: ^integer_like]

  # Actor mailbox access: the current execution context is a single actor.
  defop self(),
    results: [result: ^term_like]

  defop send(pid = ^term_like, msg = ^term_like),
    results: [result: ^term_like]

  defop receive(),
    results: [result: ^term_like]

  defop mailbox_clear(),
    results: [result: ^term_like]

  # Actor process spawn: registers a closure word as a new process entry and
  # returns its pid. The scheduler driver executes spawned entries.
  defop spawn(fun = optional(^term_like)),
    results: [result: ^term_like]

  # Explicit runtime-instance lifecycle. Hosts use these operations to give
  # each execution its own native state instead of relying on the
  # compatibility runtime associated with the invoking OS thread.
  defop runtime_create(),
    results: [result: ^integer_like]

  defop runtime_enter(handle = ^integer_like),
    results: [result: ^integer_like]

  defop runtime_leave(),
    results: [result: ^integer_like]

  defop runtime_destroy(handle = ^integer_like),
    results: [result: ^integer_like]

  # Host-result lifecycle. A result handle keeps the producing runtime alive
  # while the host materializes an arena-backed term, then releases both
  # together. Inspection always carries the handle so runtimes can reject
  # stale or foreign words instead of dereferencing arbitrary tagged i64s.
  defop result_create(runtime = ^integer_like, word = ^integer_like),
    results: [result: ^integer_like]

  # Retains a result whose unboxed i64 is known to carry an `!ex.term` word.
  # The explicit operation avoids guessing whether an immediate integer word
  # is a legacy scalar root at the host boundary.
  defop result_create_term(runtime = ^integer_like, word = ^integer_like),
    results: [result: ^integer_like]

  defop result_destroy(handle = ^integer_like),
    results: [result: ^integer_like]

  defop result_root_kind(handle = ^integer_like),
    results: [result: ^integer_like]

  defop result_root_word(handle = ^integer_like),
    results: [result: ^integer_like]

  defop result_exception_kind(handle = ^integer_like),
    results: [result: ^integer_like]

  defop result_exception_reason(handle = ^integer_like),
    results: [result: ^integer_like]

  defop result_term_kind(handle = ^integer_like, word = ^integer_like),
    results: [result: ^integer_like]

  defop result_atom_name(handle = ^integer_like, word = ^integer_like),
    results: [result: ^integer_like]

  defop result_term_length(handle = ^integer_like, word = ^integer_like),
    results: [result: ^integer_like]

  defop result_term_get(
          handle = ^integer_like,
          word = ^integer_like,
          index = ^integer_like
        ),
        results: [result: ^integer_like]

  # Portable host boundary. Exported and imported term handles are opaque
  # generation-checked capabilities; no operation exposes an arena word to
  # the host without a validating owner.
  defop term_export(result_handle = ^integer_like, word = ^integer_like),
    results: [exported_handle: ^integer_like]

  defop term_import(runtime_handle = ^integer_like, exported_handle = ^integer_like),
    results: [term_handle: ^integer_like]

  defop exported_clone(handle = ^integer_like),
    results: [result: ^integer_like]

  defop exported_destroy(handle = ^integer_like),
    results: [result: ^integer_like]

  defop exported_length(handle = ^integer_like),
    results: [result: ^integer_like]

  defop exported_get(handle = ^integer_like, index = ^integer_like),
    results: [result: ^integer_like]

  defop term_handle_export(handle = ^integer_like),
    results: [exported_handle: ^integer_like]

  defop term_handle_destroy(handle = ^integer_like),
    results: [result: ^integer_like]

  # Resets the runtime process table to a single fresh initial process with
  # the given capacity; the scheduler driver calls this at program start.
  defop process_table_reset(cap = ^integer_like),
    results: [result: ^integer_like]

  # Preemptive scheduler continuation primitives (#35 slice 5): a budgeted
  # cursor loop saves its (arg, acc, cursor) state before yielding; the
  # scheduler driver round-robins runnable processes and resumes the saved
  # continuation. A message arrival bumps the recipient's epoch, so a stale
  # continuation reads as not pending (restart from the top).
  defop cont_save(
          arg = ^integer_like,
          acc = ^integer_like,
          cursor = ^integer_like
        ),
        results: [result: ^integer_like]

  # Selective-receive scan continuation: unlike a cursor-loop continuation, a
  # message arrival invalidates it so the scan restarts (epoch wiring).
  defop receive_cont_save(
          arg = ^integer_like,
          acc = ^integer_like,
          cursor = ^integer_like
        ),
        results: [result: ^integer_like]

  defop cont_pending(),
    results: [result: ^integer_like]

  defop cont_active(),
    results: [result: ^integer_like]

  defop cont_clear(),
    results: [result: ^integer_like]

  defop cont_load_arg(),
    results: [result: ^integer_like]

  defop cont_load_acc(),
    results: [result: ^integer_like]

  defop cont_load_cursor(),
    results: [result: ^integer_like]

  defop schedule_next(),
    results: [result: ^integer_like]

  defop mailbox_len(),
    results: [result: ^integer_like]

  defop mailbox_peek(cursor = ^integer_like),
    results: [result: ^term_like]

  defop mailbox_remove(cursor = ^integer_like),
    results: [result: ^integer_like]

  defop nil_word(),
    results: [result: ^term_like]

  defop monotonic_time(),
    results: [result: ^integer_like]

  defop receive_start(),
    results: [result: ^integer_like]

  defop receive_start_set(value = ^integer_like),
    results: [result: ^integer_like]

  defop native_time(),
    results: [result: ^integer_like]

  defop unique_integer(negative = ^integer_like),
    results: [result: ^integer_like]

  defop current_entry(),
    results: [result: ^integer_like]

  defop process_done(result = ^integer_like),
    results: [result: ^integer_like]

  defop process_exit(reason = ^term_like),
    results: [result: ^term_like]

  defop process_exit_reason(pid = ^term_like),
    results: [result: ^term_like]

  defop process_trap_exit(enabled = ^integer_like),
    results: [result: ^integer_like]

  defop process_dictionary_get(key = ^term_like, default = ^term_like),
    results: [result: ^term_like]

  defop process_dictionary_put(key = ^term_like, value = ^term_like),
    results: [result: ^term_like]

  defop link(
          pid = ^term_like,
          exit_tag = ^term_like,
          normal_tag = ^term_like
        ),
        results: [result: ^term_like]

  defop unlink(pid = ^term_like),
    results: [result: ^integer_like]

  defop exit(
          pid = ^term_like,
          reason = ^term_like,
          exit_tag = ^term_like,
          normal_tag = ^term_like
        ),
        results: [result: ^term_like]

  defop monitor(
          pid = ^term_like,
          down_tag = ^term_like,
          process_tag = ^term_like,
          normal_tag = ^term_like
        ),
        results: [result: ^term_like]

  defop demonitor(reference = ^term_like),
    results: [result: ^integer_like]

  defop processes_runnable(),
    results: [result: ^integer_like]

  defop process_result(pid = ^term_like),
    results: [result: ^integer_like]

  # Parks the current actor when its mailbox has no message beyond the
  # completed selective-receive scan cursor.
  defop process_wait(cursor = ^integer_like),
    results: [result: ^integer_like]

  # Runs the actor scheduler with a fixed worker count. `dispatcher` is a
  # stable native trampoline with the signature `(pid: i64) -> i64`.
  defop worker_run(
          worker_count = ^integer_like,
          dispatcher = ^any_value
        ),
        results: [result: ^integer_like]

  # Untags an integer term word to its scalar value (a passthrough for
  # values that are already scalar).
  defop to_int(word = ^term_like),
    results: [result: ^integer_like]

  defop reduction_tick(cost = ^integer_like),
    results: [result: ^integer_like]

  defop clock_init(budget = ^integer_like),
    results: [result: ^integer_like]

  defop yield_mark(),
    results: [result: ^integer_like]

  # Non-local exit (`throw`) and its catch. The body region runs normally; a
  # throw longjmps back and the catch region matches the thrown value.
  defop try(),
    results: [result: ^any_value],
    regions: [:any, :any]

  defop throw(value = ^term_like),
    results: [result: ^term_like]

  # Raises a typed runtime exception. Unlike `throw`, this bypasses user catch
  # regions; `kind` is a runtime-defined integer discriminator and `reason`
  # carries the exception payload.
  defop raise(reason = ^term_like, kind = ^integer_like),
    results: [result: ^term_like]

  defop catch_value(),
    results: [result: ^term_like]

  # Constructs a first-class function value (a tagged closure word): the
  # extracted `__fn_*` is referenced by index, and the captured values are
  # stored in the closure's env slots.
  defop make_fun(
          e0 = optional(^any_value),
          e1 = optional(^any_value),
          e2 = optional(^any_value),
          e3 = optional(^any_value)
        ),
        results: [result: ^term_like],
        attributes: [fn_idx: ^integer_value, env_len: ^integer_value]

  # Additive closure constructor carrying the function arity. Keep
  # `ex.make_fun` available while downstream runtimes migrate from the
  # original six-word ABI.
  defop make_fun_with_arity(
          e0 = optional(^any_value),
          e1 = optional(^any_value),
          e2 = optional(^any_value),
          e3 = optional(^any_value)
        ),
        results: [result: ^term_like],
        attributes: [fn_idx: ^integer_value, arity: ^integer_value, env_len: ^integer_value]

  # Closure constructor carrying both call arity and result representation.
  # result_mode is runtime-defined (Batata uses 0 for scalar, 1 for term).
  defop make_fun_with_signature(
          e0 = optional(^any_value),
          e1 = optional(^any_value),
          e2 = optional(^any_value),
          e3 = optional(^any_value)
        ),
        results: [result: ^term_like],
        attributes: [
          fn_idx: ^integer_value,
          arity: ^integer_value,
          result_mode: ^integer_value,
          env_len: ^integer_value
        ]

  # Reads a closure's declared arity. Runtimes return -1 for values or legacy
  # closures which do not carry arity metadata.
  defop fun_arity(fun = ^term_like),
    results: [result: ^integer_like]

  # Reads the runtime-defined result representation carried by a closure.
  defop fun_result_mode(fun = ^term_like),
    results: [result: ^integer_like]

  # Applies a first-class function value. The closure is resolved at runtime
  # to its function index and env slots, then dispatched to the matching
  # `__fn_*`.
  defop apply(
          closure = optional(^term_like),
          a0 = optional(^any_value),
          a1 = optional(^any_value),
          a2 = optional(^any_value),
          a3 = optional(^any_value)
        ),
        results: [result: ^any_value],
        attributes: [arg_count: ^integer_value]

  defop tuple(elements = variadic(^term_like)),
    results: [result: ^term_like]

  defop list(elements = variadic(^term_like)),
    results: [result: ^term_like]

  defop list_cons(head = ^term_like, tail = ^term_like),
    results: [result: ^term_like]

  defop list_flatten(list = ^term_like),
    results: [result: ^term_like]

  defop list_insert_at(list = ^term_like, index = ^integer_like, value = ^term_like),
    results: [result: ^term_like]

  defop map(entries = variadic(^term_like)),
    results: [result: ^term_like]

  defop map_put(map = ^term_like, key = ^term_like, value = ^term_like),
    results: [result: ^term_like]

  defop map_fetch(map = ^term_like, key = ^term_like),
    results: [result: ^term_like]

  defop mapset_from_list(list = ^term_like),
    results: [result: ^term_like]

  defop mapset_member(set = ^term_like, member = ^term_like),
    results: [result: ^integer_like]

  defop mapset_put(set = ^term_like, member = ^term_like),
    results: [result: ^term_like]

  defop file_read(path = ^term_like),
    results: [result: ^term_like]

  defop file_read_lines(path = ^term_like),
    results: [result: ^term_like]

  defop binary(segments = variadic(^term_like)),
    results: [result: ^term_like]

  defop binary_from_list(bytes = ^term_like),
    results: [result: ^term_like]

  defop iodata_to_binary(iodata = ^term_like),
    results: [result: ^term_like]

  defop float_lit(bits = ^integer_like),
    results: [result: ^term_like]

  defop bigint_lit(decimal = ^term_like),
    results: [result: ^term_like]

  defop string_to_float(binary = ^term_like),
    results: [result: ^term_like]

  defop string_to_atom(binary = ^term_like),
    results: [result: ^term_like]

  defop string_to_existing_atom(binary = ^term_like),
    results: [result: ^term_like]

  defop float_to_binary_short(value = ^term_like),
    results: [result: ^term_like]

  defop is_integer(value = ^any_value),
    results: [result: ^integer_like]

  defop is_float(value = ^any_value),
    results: [result: ^integer_like]

  defop is_atom(value = ^any_value),
    results: [result: ^integer_like]

  defop is_binary(value = ^any_value),
    results: [result: ^integer_like]

  defop is_list(value = ^any_value),
    results: [result: ^integer_like]

  defop is_tuple(value = ^any_value),
    results: [result: ^integer_like]

  defop is_map(value = ^any_value),
    results: [result: ^integer_like]

  defop tuple_get(tuple = ^term_like, index = ^integer_like),
    results: [result: ^term_like]

  defop tuple_length(tuple = ^term_like),
    results: [result: ^integer_like]

  defop map_length(map = ^term_like),
    results: [result: ^integer_like]

  defop enumerable_count(word = ^term_like),
    results: [result: ^integer_like]

  defop enumerable_to_list(word = ^term_like),
    results: [result: ^term_like]

  defop enumerable_into_map(enumerable = ^term_like, target = ^term_like),
    results: [result: ^term_like]

  defop enumerable_intersperse(enumerable = ^term_like, separator = ^term_like),
    results: [result: ^term_like]

  defop enumerable_to_list_range(
          start = ^integer_like,
          stop = ^integer_like
        ),
        results: [result: ^term_like]

  defop enumerable_reduce(
          enumerable = ^term_like,
          acc = ^integer_like,
          continuation = ^integer_like
        ),
        results: [result: ^integer_like]

  defop enumerable_reduce_c(
          enumerable = ^term_like,
          acc = ^integer_like,
          continuation = ^integer_like,
          capture = ^integer_like
        ),
        results: [result: ^integer_like]

  defop enumerable_reduce_range(
          start = ^integer_like,
          stop = ^integer_like,
          acc = ^integer_like,
          continuation = ^integer_like
        ),
        results: [result: ^integer_like]

  defop enumerable_reduce_fun(
          enumerable = ^term_like,
          acc = ^integer_like,
          reducer = ^any_value
        ),
        results: [result: ^integer_like]

  defop enumerable_map_fun(
          enumerable = ^term_like,
          mapper = ^any_value
        ),
        results: [result: ^term_like]

  defop enumerable_map_term_fun(
          enumerable = ^term_like,
          mapper = ^any_value
        ),
        results: [result: ^term_like]

  defop enumerable_map_term_fun_c(
          enumerable = ^term_like,
          mapper = ^any_value,
          env0 = ^any_value,
          env1 = ^any_value,
          env2 = ^any_value,
          env3 = ^any_value
        ),
        results: [result: ^term_like]

  defop enumerable_flat_map_term_fun(
          enumerable = ^term_like,
          mapper = ^any_value
        ),
        results: [result: ^term_like]

  defop stream_filter(
          list = ^term_like,
          predicate = ^any_value
        ),
        results: [result: ^term_like]

  defop stream_take(
          list = ^term_like,
          n = ^integer_like
        ),
        results: [result: ^term_like]

  defop stream_drop(
          list = ^term_like,
          n = ^integer_like
        ),
        results: [result: ^term_like]

  defop func_addr(),
    attributes: [sym_name: base("#builtin.string")],
    results: [result: ^any_value]

  defop list_head(list = ^term_like),
    results: [result: ^term_like]

  defop list_tail(list = ^term_like),
    results: [result: ^term_like]

  defop list_get(list = ^term_like, index = ^integer_like),
    results: [result: ^term_like]

  defop list_length(list = ^term_like),
    results: [result: ^integer_like]

  defop term_eq(left = ^term_like, right = ^term_like),
    results: [result: ^integer_like]

  defop term_eq_loose(left = ^term_like, right = ^term_like),
    results: [result: ^integer_like]

  # Exact integer ordering over tagged immediate or boxed integer terms.
  # The runtime returns -1, 0, or 1; callers retain responsibility for
  # validating that both operands are integers when guard semantics require it.
  defop integer_compare(left = ^term_like, right = ^term_like),
    results: [result: ^integer_like]

  # Representation-preserving integer arithmetic. Unlike the scalar ex.add/
  # sub/mul/div/rem operations, these accept tagged immediate or boxed integer
  # terms and return a term so arbitrary precision is not narrowed through i64.
  defop integer_add(left = ^term_like, right = ^term_like),
    results: [result: ^term_like]

  defop integer_sub(left = ^term_like, right = ^term_like),
    results: [result: ^term_like]

  defop integer_mul(left = ^term_like, right = ^term_like),
    results: [result: ^term_like]

  defop integer_div(left = ^term_like, right = ^term_like),
    results: [result: ^term_like]

  defop integer_rem(left = ^term_like, right = ^term_like),
    results: [result: ^term_like]

  defop binary_length(binary = ^term_like),
    results: [result: ^integer_like]

  defop binary_get(binary = ^term_like, index = ^integer_like),
    results: [result: ^term_like]

  defop binary_slice(binary = ^term_like, start = ^integer_like),
    results: [result: ^term_like]

  defop binary_part(
          binary = ^term_like,
          start = ^term_like,
          length = ^term_like
        ),
        results: [result: ^term_like]

  defop binary_utf8_get(binary = ^term_like, index = ^integer_like),
    results: [result: ^term_like]

  defop binary_utf8_width(
          binary = ^term_like,
          index = ^integer_like
        ),
        results: [result: ^integer_like]

  defop binary_utf8_length(binary = ^term_like),
    results: [result: ^integer_like]

  defop string_printable(binary = ^term_like),
    results: [result: ^integer_like]

  defop binary_quote(binary = ^term_like),
    results: [result: ^term_like]

  defop binary_encode16(binary = ^term_like),
    results: [result: ^term_like]

  defop binary_decode16(binary = ^term_like),
    results: [result: ^term_like]

  defop int_to_string(word = ^term_like),
    results: [result: ^term_like]

  defop int_to_string_base(word = ^term_like, base = ^integer_like),
    results: [result: ^term_like]

  defop int_to_hex(word = ^term_like),
    results: [result: ^term_like]

  defop string_to_int(binary = ^term_like),
    results: [result: ^integer_like]

  defop yield(values = variadic(^any_value)), traits: [:terminator]

  defop func(),
    attributes: [sym_name: base("#builtin.string")],
    regions: [body: {:region, size: 1}],
    traits: [:isolated_from_above]

  defop return(value = optional(^any_value)), traits: [:terminator]
end
