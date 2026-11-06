defmodule Beaver.MLIR.Dialect.Ex.Runtime do
  @moduledoc "Tagged-word representation, closure layout and host-handle operations."

  @doc false
  defmacro __using__(_) do
    quote do
      @ex_layer :runtime

      defop box(value = ^any_value), results: [result: ^term_like]

      # Reinterprets an already-tagged word as a term without tagging.
      defop to_word(value = ^any_value), results: [result: ^term_like]

      # Drops the term annotation without untagging, for scalar i64 control-flow edges.
      defop unbox(word = ^term_like), results: [result: ^integer_like]

      # Explicit runtime lifecycle avoids the thread-local compatibility runtime.
      defops ~w(runtime_create runtime_leave native_time)a, [], results: [result: ^integer_like]

      defops ~w(
        runtime_enter runtime_destroy result_destroy result_root_kind result_root_word result_exception_kind
        result_exception_reason exported_clone exported_destroy exported_length term_handle_destroy
      )a,
             [handle = ^integer_like], results: [result: ^integer_like]

      # Result handles retain their runtime/arena while the host reads terms.
      # Inspection validates ownership before dereferencing tagged words.
      defop result_create(runtime = ^integer_like, word = ^integer_like), results: [result: ^integer_like]

      # Marks an i64 root as an ex.term word instead of guessing its representation.
      defop result_create_term(runtime = ^integer_like, word = ^integer_like), results: [result: ^integer_like]

      defops ~w(result_term_kind result_atom_name result_term_length)a, [handle = ^integer_like, word = ^integer_like],
        results: [result: ^integer_like]

      defop result_term_get(handle = ^integer_like, word = ^integer_like, index = ^integer_like),
        results: [result: ^integer_like]

      # Export/import handles are generation-checked capabilities that validate ownership.
      defop term_export(result_handle = ^integer_like, word = ^integer_like), results: [exported_handle: ^integer_like]

      defop term_import(runtime_handle = ^integer_like, exported_handle = ^integer_like),
        results: [term_handle: ^integer_like]

      defop exported_get(handle = ^integer_like, index = ^integer_like), results: [result: ^integer_like]

      defop term_handle_export(handle = ^integer_like), results: [exported_handle: ^integer_like]

      defop nil_word(), results: [result: ^term_like]

      # Untags an integer term word; already-scalar values pass through.
      defop to_int(word = ^term_like), results: [result: ^integer_like]

      # Tagged closure: function index plus at most four captured environment words.
      defop make_fun(
              e0 = optional(^any_value),
              e1 = optional(^any_value),
              e2 = optional(^any_value),
              e3 = optional(^any_value)
            ),
            results: [result: ^term_like],
            attributes: [fn_idx: ^integer_value, env_len: ^integer_value]

      # Adds call arity to the original six-word closure ABI.
      defop make_fun_with_arity(
              e0 = optional(^any_value),
              e1 = optional(^any_value),
              e2 = optional(^any_value),
              e3 = optional(^any_value)
            ),
            results: [result: ^term_like],
            attributes: [fn_idx: ^integer_value, arity: ^integer_value, env_len: ^integer_value]

      # Adds result_mode (Batata: 0 = scalar, 1 = term) to call arity.
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

      # Dispatches the closure's function index with its environment and arguments.
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
