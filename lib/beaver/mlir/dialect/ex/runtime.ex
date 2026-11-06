defmodule Beaver.MLIR.Dialect.Ex.Runtime do
  @moduledoc "Tagged-word representation, closure layout and host-handle operations."

  @doc false
  defmacro __using__(_) do
    # Slang evaluates this owner's declarations together in the Ex facade.
    # credo:disable-for-next-line Credo.Check.Refactor.LongQuoteBlocks
    quote do
      @ex_layer :runtime

      defop box(value = ^any_value), results: [result: ^term_like]

      # Lifts an already-tagged word (e.g. a function argument) into the term
      # type without tagging: the conversion is a pure passthrough.
      defop to_word(value = ^any_value), results: [result: ^term_like]

      # Drops the term type annotation without untagging: the conversion is a
      # pure passthrough, so a term word can cross control-flow regions (whose
      # types must be legal after conversion) as a scalar i64.
      defop unbox(word = ^term_like), results: [result: ^integer_like]

      # Explicit runtime-instance lifecycle. Hosts use these operations to give
      # each execution its own native state instead of relying on the
      # compatibility runtime associated with the invoking OS thread.
      defop runtime_create(), results: [result: ^integer_like]

      defop runtime_enter(handle = ^integer_like), results: [result: ^integer_like]

      defop runtime_leave(), results: [result: ^integer_like]

      defop runtime_destroy(handle = ^integer_like), results: [result: ^integer_like]

      # Host-result lifecycle. A result handle keeps the producing runtime alive
      # while the host materializes an arena-backed term, then releases both
      # together. Inspection always carries the handle so runtimes can reject
      # stale or foreign words instead of dereferencing arbitrary tagged i64s.
      defop result_create(runtime = ^integer_like, word = ^integer_like), results: [result: ^integer_like]

      # Retains a result whose unboxed i64 is known to carry an `!ex.term` word.
      # The explicit operation avoids guessing whether an immediate integer word
      # is a legacy scalar root at the host boundary.
      defop result_create_term(runtime = ^integer_like, word = ^integer_like), results: [result: ^integer_like]

      defop result_destroy(handle = ^integer_like), results: [result: ^integer_like]

      defop result_root_kind(handle = ^integer_like), results: [result: ^integer_like]

      defop result_root_word(handle = ^integer_like), results: [result: ^integer_like]

      defop result_exception_kind(handle = ^integer_like), results: [result: ^integer_like]

      defop result_exception_reason(handle = ^integer_like), results: [result: ^integer_like]

      defop result_term_kind(handle = ^integer_like, word = ^integer_like), results: [result: ^integer_like]

      defop result_atom_name(handle = ^integer_like, word = ^integer_like), results: [result: ^integer_like]

      defop result_term_length(handle = ^integer_like, word = ^integer_like), results: [result: ^integer_like]

      defop result_term_get(handle = ^integer_like, word = ^integer_like, index = ^integer_like),
        results: [result: ^integer_like]

      # Portable host boundary. Exported and imported term handles are opaque
      # generation-checked capabilities; no operation exposes an arena word to
      # the host without a validating owner.
      defop term_export(result_handle = ^integer_like, word = ^integer_like), results: [exported_handle: ^integer_like]

      defop term_import(runtime_handle = ^integer_like, exported_handle = ^integer_like),
        results: [term_handle: ^integer_like]

      defop exported_clone(handle = ^integer_like), results: [result: ^integer_like]

      defop exported_destroy(handle = ^integer_like), results: [result: ^integer_like]

      defop exported_length(handle = ^integer_like), results: [result: ^integer_like]

      defop exported_get(handle = ^integer_like, index = ^integer_like), results: [result: ^integer_like]

      defop term_handle_export(handle = ^integer_like), results: [exported_handle: ^integer_like]

      defop term_handle_destroy(handle = ^integer_like), results: [result: ^integer_like]

      defop nil_word(), results: [result: ^term_like]

      defop native_time(), results: [result: ^integer_like]

      # Untags an integer term word to its scalar value (a passthrough for
      # values that are already scalar).
      defop to_int(word = ^term_like), results: [result: ^integer_like]

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

      # Reads the runtime-defined result representation carried by a closure.
      defop fun_result_mode(fun = ^term_like), results: [result: ^integer_like]

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

      defop float_lit(bits = ^integer_like), results: [result: ^term_like]

      defop bigint_lit(decimal = ^term_like), results: [result: ^term_like]

      defop enumerable_reduce_c(
              enumerable = ^term_like,
              acc = ^integer_like,
              continuation = ^integer_like,
              capture = ^integer_like
            ),
            results: [result: ^integer_like]

      defop enumerable_map_term_fun_c(
              enumerable = ^term_like,
              mapper = ^any_value,
              env0 = ^any_value,
              env1 = ^any_value,
              env2 = ^any_value,
              env3 = ^any_value
            ),
            results: [result: ^term_like]

      defop func_addr(), attributes: [sym_name: base("#builtin.string")], results: [result: ^any_value]
    end
  end
end
