defmodule Beaver.MLIR.Dialect.Ex.Language do
  @moduledoc "Elixir values, calls and structured control; existing scalar and clause restrictions remain."

  @doc false
  defmacro __using__(_) do
    # Slang evaluates this owner's declarations together in the Ex facade.
    # credo:disable-for-next-line Credo.Check.Refactor.LongQuoteBlocks
    quote do
      @ex_layer :language

      defop lit(), results: [result: ^integer_like], attributes: [value: ^integer_value]

      defop var(), results: [result: base(unbound())], attributes: [name: base("#builtin.string")]

      defop bind(variable = base(unbound()), value = ^any_value), results: [result: base(bound())]

      defop add(left = ^integer_like, right = ^integer_like), results: [result: ^integer_like]

      defop sub(left = ^integer_like, right = ^integer_like), results: [result: ^integer_like]

      defop mul(left = ^integer_like, right = ^integer_like), results: [result: ^integer_like]

      defop div(left = ^integer_like, right = ^integer_like), results: [result: ^integer_like]

      defop rem(left = ^integer_like, right = ^integer_like), results: [result: ^integer_like]

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

      defop cmp(left = ^integer_like, right = ^integer_like),
        results: [result: ^integer_like],
        attributes: [predicate: ^cmp_predicate_value]

      # credo:disable-for-next-line Credo.Check.Readability.ParenthesesInCondition
      defop if(cond = ^integer_like),
        results: [result: variadic(^any_value)],
        regions: [:any, :any]

      defop case(scrutinee = ^any_value), results: [result: variadic(^any_value)], regions: [:any]

      defop clause(guard = optional(^any_value)), attributes: [patterns: ^any_value]

      # Actor mailbox access: the current execution context is a single actor.
      defop self(), results: [result: ^term_like]

      defop send(pid = ^term_like, msg = ^term_like), results: [result: ^term_like]

      defop receive(), results: [result: ^term_like]

      # Actor process spawn: registers a closure word as a new process entry and
      # returns its pid. The scheduler driver executes spawned entries.
      defop spawn(fun = optional(^term_like)), results: [result: ^term_like]

      defop unique_integer(negative = ^integer_like), results: [result: ^integer_like]

      defop process_exit(reason = ^term_like), results: [result: ^term_like]

      defop process_exit_reason(pid = ^term_like), results: [result: ^term_like]

      defop process_trap_exit(enabled = ^integer_like), results: [result: ^integer_like]

      defop process_dictionary_get(key = ^term_like, default = ^term_like), results: [result: ^term_like]

      defop process_dictionary_put(key = ^term_like, value = ^term_like), results: [result: ^term_like]

      defop link(pid = ^term_like, exit_tag = ^term_like, normal_tag = ^term_like), results: [result: ^term_like]

      defop unlink(pid = ^term_like), results: [result: ^integer_like]

      defop exit(pid = ^term_like, reason = ^term_like, exit_tag = ^term_like, normal_tag = ^term_like),
        results: [result: ^term_like]

      defop monitor(pid = ^term_like, down_tag = ^term_like, process_tag = ^term_like, normal_tag = ^term_like),
        results: [result: ^term_like]

      defop demonitor(reference = ^term_like), results: [result: ^integer_like]

      # Non-local exit (`throw`) and its catch. The body region runs normally; a
      # throw longjmps back and the catch region matches the thrown value.
      defop try(),
        results: [result: ^any_value],
        regions: [:any, :any]

      defop throw(value = ^term_like), results: [result: ^term_like]

      # Raises a typed runtime exception. Unlike `throw`, this bypasses user catch
      # regions; `kind` is a runtime-defined integer discriminator and `reason`
      # carries the exception payload.
      defop raise(reason = ^term_like, kind = ^integer_like), results: [result: ^term_like]

      defop catch_value(), results: [result: ^term_like]

      # Reads a closure's declared arity. Runtimes return -1 for values or legacy
      # closures which do not carry arity metadata.
      defop fun_arity(fun = ^term_like), results: [result: ^integer_like]

      defop tuple(elements = variadic(^term_like)), results: [result: ^term_like]

      defop list(elements = variadic(^term_like)), results: [result: ^term_like]

      defop list_cons(head = ^term_like, tail = ^term_like), results: [result: ^term_like]

      defop list_flatten(list = ^term_like), results: [result: ^term_like]

      defop list_insert_at(list = ^term_like, index = ^integer_like, value = ^term_like), results: [result: ^term_like]

      defop map(entries = variadic(^term_like)), results: [result: ^term_like]

      defop map_put(map = ^term_like, key = ^term_like, value = ^term_like), results: [result: ^term_like]

      defop map_fetch(map = ^term_like, key = ^term_like), results: [result: ^term_like]

      defop mapset_from_list(list = ^term_like), results: [result: ^term_like]

      defop mapset_member(set = ^term_like, member = ^term_like), results: [result: ^integer_like]

      defop mapset_put(set = ^term_like, member = ^term_like), results: [result: ^term_like]

      defop file_read(path = ^term_like), results: [result: ^term_like]

      defop file_read_lines(path = ^term_like), results: [result: ^term_like]

      defop binary(segments = variadic(^term_like)), results: [result: ^term_like]

      defop binary_from_list(bytes = ^term_like), results: [result: ^term_like]

      defop iodata_to_binary(iodata = ^term_like), results: [result: ^term_like]

      defop string_to_float(binary = ^term_like), results: [result: ^term_like]

      defop string_to_atom(binary = ^term_like), results: [result: ^term_like]

      defop string_to_existing_atom(binary = ^term_like), results: [result: ^term_like]

      defop float_to_binary_short(value = ^term_like), results: [result: ^term_like]

      defop is_integer(value = ^any_value), results: [result: ^integer_like]

      defop is_float(value = ^any_value), results: [result: ^integer_like]

      defop is_atom(value = ^any_value), results: [result: ^integer_like]

      defop is_binary(value = ^any_value), results: [result: ^integer_like]

      defop is_list(value = ^any_value), results: [result: ^integer_like]

      defop is_tuple(value = ^any_value), results: [result: ^integer_like]

      defop is_map(value = ^any_value), results: [result: ^integer_like]

      defop tuple_get(tuple = ^term_like, index = ^integer_like), results: [result: ^term_like]

      defop tuple_length(tuple = ^term_like), results: [result: ^integer_like]

      defop map_length(map = ^term_like), results: [result: ^integer_like]

      defop enumerable_count(word = ^term_like), results: [result: ^integer_like]

      defop enumerable_to_list(word = ^term_like), results: [result: ^term_like]

      defop enumerable_into_map(enumerable = ^term_like, target = ^term_like), results: [result: ^term_like]

      defop enumerable_intersperse(enumerable = ^term_like, separator = ^term_like), results: [result: ^term_like]

      defop enumerable_to_list_range(start = ^integer_like, stop = ^integer_like), results: [result: ^term_like]

      defop enumerable_reduce(enumerable = ^term_like, acc = ^integer_like, continuation = ^integer_like),
        results: [result: ^integer_like]

      defop enumerable_reduce_range(
              start = ^integer_like,
              stop = ^integer_like,
              acc = ^integer_like,
              continuation = ^integer_like
            ),
            results: [result: ^integer_like]

      defop enumerable_reduce_fun(enumerable = ^term_like, acc = ^integer_like, reducer = ^any_value),
        results: [result: ^integer_like]

      defop enumerable_map_fun(enumerable = ^term_like, mapper = ^any_value), results: [result: ^term_like]

      defop enumerable_map_term_fun(enumerable = ^term_like, mapper = ^any_value), results: [result: ^term_like]

      defop enumerable_flat_map_term_fun(enumerable = ^term_like, mapper = ^any_value), results: [result: ^term_like]

      defop stream_filter(list = ^term_like, predicate = ^any_value), results: [result: ^term_like]

      defop stream_take(list = ^term_like, n = ^integer_like), results: [result: ^term_like]

      defop stream_drop(list = ^term_like, n = ^integer_like), results: [result: ^term_like]

      defop list_head(list = ^term_like), results: [result: ^term_like]

      defop list_tail(list = ^term_like), results: [result: ^term_like]

      defop list_get(list = ^term_like, index = ^integer_like), results: [result: ^term_like]

      defop list_length(list = ^term_like), results: [result: ^integer_like]

      defop term_eq(left = ^term_like, right = ^term_like), results: [result: ^integer_like]

      defop term_eq_loose(left = ^term_like, right = ^term_like), results: [result: ^integer_like]

      # Exact integer ordering over tagged immediate or boxed integer terms.
      # The runtime returns -1, 0, or 1; callers retain responsibility for
      # validating that both operands are integers when guard semantics require it.
      defop integer_compare(left = ^term_like, right = ^term_like), results: [result: ^integer_like]

      # Representation-preserving integer arithmetic. Unlike the scalar ex.add/
      # sub/mul/div/rem operations, these accept tagged immediate or boxed integer
      # terms and return a term so arbitrary precision is not narrowed through i64.
      defop integer_add(left = ^term_like, right = ^term_like), results: [result: ^term_like]

      defop integer_sub(left = ^term_like, right = ^term_like), results: [result: ^term_like]

      defop integer_mul(left = ^term_like, right = ^term_like), results: [result: ^term_like]

      defop integer_div(left = ^term_like, right = ^term_like), results: [result: ^term_like]

      defop integer_rem(left = ^term_like, right = ^term_like), results: [result: ^term_like]

      defop binary_length(binary = ^term_like), results: [result: ^integer_like]

      defop binary_get(binary = ^term_like, index = ^integer_like), results: [result: ^term_like]

      defop binary_slice(binary = ^term_like, start = ^integer_like), results: [result: ^term_like]

      defop binary_part(binary = ^term_like, start = ^term_like, length = ^term_like), results: [result: ^term_like]

      defop binary_utf8_get(binary = ^term_like, index = ^integer_like), results: [result: ^term_like]

      defop binary_utf8_width(binary = ^term_like, index = ^integer_like), results: [result: ^integer_like]

      defop binary_utf8_length(binary = ^term_like), results: [result: ^integer_like]

      defop string_printable(binary = ^term_like), results: [result: ^integer_like]

      defop binary_quote(binary = ^term_like), results: [result: ^term_like]

      defop binary_encode16(binary = ^term_like), results: [result: ^term_like]

      defop binary_decode16(binary = ^term_like), results: [result: ^term_like]

      defop int_to_string(word = ^term_like), results: [result: ^term_like]

      defop int_to_string_base(word = ^term_like, base = ^integer_like), results: [result: ^term_like]

      defop int_to_hex(word = ^term_like), results: [result: ^term_like]

      defop string_to_int(binary = ^term_like), results: [result: ^integer_like]

      defop yield(values = variadic(^any_value)), traits: [:terminator]

      defop func(),
        attributes: [sym_name: base("#builtin.string")],
        regions: [body: {:region, size: 1}],
        traits: [:isolated_from_above]

      defop return(value = optional(^any_value)), traits: [:terminator]
    end
  end
end
