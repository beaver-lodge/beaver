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

      defops ~w(add sub mul div rem)a, [left = ^integer_like, right = ^integer_like], results: [result: ^integer_like]

      # Eight optional slots permit four captures plus four heterogeneous arguments;
      # a single IRDL variadic group would require homogeneous types.
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

      # Registers a closure as a process; the scheduler executes it.
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

      # The body runs normally; throw longjmps into the catch region.
      defop try(),
        results: [result: ^any_value],
        regions: [:any, :any]

      defop throw(value = ^term_like), results: [result: ^term_like]

      # Typed runtime exception bypassing user catch; kind is runtime-defined.
      defop raise(reason = ^term_like, kind = ^integer_like), results: [result: ^term_like]

      defop catch_value(), results: [result: ^term_like]

      # Returns declared arity, or -1 when the runtime has no arity metadata.
      defop fun_arity(fun = ^term_like), results: [result: ^integer_like]

      defops ~w(tuple list)a, [elements = variadic(^term_like)], results: [result: ^term_like]

      defop list_cons(head = ^term_like, tail = ^term_like), results: [result: ^term_like]

      defops ~w(list_flatten mapset_from_list list_head list_tail)a, [list = ^term_like], results: [result: ^term_like]

      defop list_insert_at(list = ^term_like, index = ^integer_like, value = ^term_like), results: [result: ^term_like]

      defop map(entries = variadic(^term_like)), results: [result: ^term_like]
      defop map_put(map = ^term_like, key = ^term_like, value = ^term_like), results: [result: ^term_like]
      defop map_fetch(map = ^term_like, key = ^term_like), results: [result: ^term_like]

      defop mapset_member(set = ^term_like, member = ^term_like), results: [result: ^integer_like]
      defop mapset_put(set = ^term_like, member = ^term_like), results: [result: ^term_like]

      defops ~w(file_read file_read_lines)a, [path = ^term_like], results: [result: ^term_like]

      defop binary(segments = variadic(^term_like)), results: [result: ^term_like]

      defop binary_from_list(bytes = ^term_like), results: [result: ^term_like]

      defop iodata_to_binary(iodata = ^term_like), results: [result: ^term_like]

      defops ~w(string_to_float string_to_atom string_to_existing_atom binary_quote binary_encode16 binary_decode16)a,
             [binary = ^term_like], results: [result: ^term_like]

      defop float_to_binary_short(value = ^term_like), results: [result: ^term_like]

      defops ~w(is_integer is_float is_atom is_binary is_list is_tuple is_map)a, [value = ^any_value],
        results: [result: ^integer_like]

      defop tuple_get(tuple = ^term_like, index = ^integer_like), results: [result: ^term_like]
      defop tuple_length(tuple = ^term_like), results: [result: ^integer_like]

      defop map_length(map = ^term_like), results: [result: ^integer_like]

      defop enumerable_count(word = ^term_like), results: [result: ^integer_like]

      defops ~w(enumerable_to_list int_to_string int_to_hex)a, [word = ^term_like], results: [result: ^term_like]

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

      defops ~w(enumerable_map_fun enumerable_map_term_fun enumerable_flat_map_term_fun)a,
             [enumerable = ^term_like, mapper = ^any_value], results: [result: ^term_like]

      defop stream_filter(list = ^term_like, predicate = ^any_value), results: [result: ^term_like]

      defops ~w(stream_take stream_drop)a, [list = ^term_like, n = ^integer_like], results: [result: ^term_like]

      defop list_get(list = ^term_like, index = ^integer_like), results: [result: ^term_like]
      defop list_length(list = ^term_like), results: [result: ^integer_like]

      # integer_compare returns -1/0/1; guards must validate integer operands first.
      defops ~w(term_eq term_eq_loose integer_compare)a, [left = ^term_like, right = ^term_like],
        results: [result: ^integer_like]

      # Term arithmetic preserves arbitrary precision across immediate and boxed integers.
      defops ~w(integer_add integer_sub integer_mul integer_div integer_rem)a, [left = ^term_like, right = ^term_like],
        results: [result: ^term_like]

      defops ~w(binary_length binary_utf8_length string_printable string_to_int)a, [binary = ^term_like],
        results: [result: ^integer_like]

      defops ~w(binary_get binary_utf8_get)a, [binary = ^term_like, index = ^integer_like],
        results: [result: ^term_like]

      defop binary_slice(binary = ^term_like, start = ^integer_like), results: [result: ^term_like]
      defop binary_part(binary = ^term_like, start = ^term_like, length = ^term_like), results: [result: ^term_like]
      defop binary_utf8_width(binary = ^term_like, index = ^integer_like), results: [result: ^integer_like]

      defop int_to_string_base(word = ^term_like, base = ^integer_like), results: [result: ^term_like]

      defop yield(values = variadic(^any_value)), traits: [:terminator]

      defop func(),
        attributes: [sym_name: base("#builtin.string")],
        regions: [body: {:region, size: 1}],
        traits: [:isolated_from_above]

      defop return(value = optional(^any_value)), traits: [:terminator]
    end
  end
end
