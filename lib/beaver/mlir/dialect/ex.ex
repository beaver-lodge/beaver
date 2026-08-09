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

  use Beaver.Slang, name: "ex"

  deftype dyn()
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

  defop lit(),
    results: [result: all_of([base("!builtin.integer"), any()])],
    attributes: [value: ^integer_value]

  defop var(),
    results: [result: base(unbound())],
    attributes: [name: base("#builtin.string")]

  defop bind(variable = base(unbound()), value = any()),
    results: [result: base(bound())]

  defop add(
          left = all_of([base("!builtin.integer"), any()]),
          right = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: all_of([base("!builtin.integer"), any()])]

  defop sub(
          left = all_of([base("!builtin.integer"), any()]),
          right = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: all_of([base("!builtin.integer"), any()])]

  defop mul(
          left = all_of([base("!builtin.integer"), any()]),
          right = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: all_of([base("!builtin.integer"), any()])]

  # Each argument is its own optional slot so heterogeneous argument types
  # (e.g. a term plus a scalar accumulator) verify: IRDL variadic groups are
  # homogeneous, which would reject mixed-typed calls. Eight slots cover the
  # closure ABI: four captured values plus four application arguments.
  defop call(
          arg0 = optional(any()),
          arg1 = optional(any()),
          arg2 = optional(any()),
          arg3 = optional(any()),
          arg4 = optional(any()),
          arg5 = optional(any()),
          arg6 = optional(any()),
          arg7 = optional(any())
        ),
        results: [result: any()],
        attributes: [callee: ^callee_value, arity: ^integer_value]

  defop cmp(
          left = all_of([base("!builtin.integer"), any()]),
          right = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: all_of([base("!builtin.integer"), any()])],
        attributes: [predicate: ^cmp_predicate_value]

  # credo:disable-for-next-line Credo.Check.Readability.ParenthesesInCondition
  defop if(cond = all_of([base("!builtin.integer"), any()])),
    results: [result: variadic(any())],
    regions: [:any, :any]

  defop case(scrutinee = any()),
    results: [result: variadic(any())],
    regions: [:any]

  defop clause(guard = optional(any())),
    attributes: [patterns: any()]

  defop box(value = any()),
    results: [result: base(dyn())]

  # Lifts an already-tagged word (e.g. a function argument) into the term
  # type without tagging: the conversion is a pure passthrough.
  defop to_word(value = any()),
    results: [result: base(dyn())]

  # Drops the term type annotation without untagging: the conversion is a
  # pure passthrough, so a term word can cross control-flow regions (whose
  # types must be legal after conversion) as a scalar i64.
  defop unbox(word = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  # Actor mailbox access: the current execution context is a single actor.
  defop self(),
    results: [result: base(dyn())]

  defop send(pid = base(dyn()), msg = base(dyn())),
    results: [result: base(dyn())]

  defop receive(),
    results: [result: base(dyn())]

  defop mailbox_clear(),
    results: [result: base(dyn())]

  # Untags an integer term word to its scalar value (a passthrough for
  # values that are already scalar).
  defop to_int(word = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  # Non-local exit (`throw`) and its catch. The body region runs normally; a
  # throw longjmps back and the catch region matches the thrown value.
  defop try(),
    results: [result: any()],
    regions: [:any, :any]

  defop throw(value = base(dyn())),
    results: [result: base(dyn())]

  defop catch_value(),
    results: [result: base(dyn())]

  # Constructs a first-class function value (a tagged closure word): the
  # extracted `__fn_*` is referenced by index, and the captured values are
  # stored in the closure's env slots.
  defop make_fun(
          e0 = optional(any()),
          e1 = optional(any()),
          e2 = optional(any()),
          e3 = optional(any())
        ),
        results: [result: base(dyn())],
        attributes: [fn_idx: ^integer_value, env_len: ^integer_value]

  # Applies a first-class function value. The closure is resolved at runtime
  # to its function index and env slots, then dispatched to the matching
  # `__fn_*`.
  defop apply(
          closure = optional(base(dyn())),
          a0 = optional(any()),
          a1 = optional(any()),
          a2 = optional(any()),
          a3 = optional(any())
        ),
        results: [result: any()],
        attributes: [arg_count: ^integer_value]

  defop tuple(elements = variadic(base(dyn()))),
    results: [result: base(dyn())]

  defop list(elements = variadic(base(dyn()))),
    results: [result: base(dyn())]

  defop list_cons(head = base(dyn()), tail = base(dyn())),
    results: [result: base(dyn())]

  defop map(entries = variadic(base(dyn()))),
    results: [result: base(dyn())]

  defop binary(segments = variadic(base(dyn()))),
    results: [result: base(dyn())]

  defop is_integer(value = any()),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop is_atom(value = any()),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop is_binary(value = any()),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop is_list(value = any()),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop is_tuple(value = any()),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop is_map(value = any()),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop tuple_get(tuple = base(dyn()), index = all_of([base("!builtin.integer"), any()])),
    results: [result: base(dyn())]

  defop tuple_length(tuple = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop map_length(map = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop enumerable_count(word = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop enumerable_to_list(word = base(dyn())),
    results: [result: base(dyn())]

  defop enumerable_to_list_range(
          start = all_of([base("!builtin.integer"), any()]),
          stop = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: base(dyn())]

  defop enumerable_reduce(
          enumerable = base(dyn()),
          acc = all_of([base("!builtin.integer"), any()]),
          continuation = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: all_of([base("!builtin.integer"), any()])]

  defop enumerable_reduce_c(
          enumerable = base(dyn()),
          acc = all_of([base("!builtin.integer"), any()]),
          continuation = all_of([base("!builtin.integer"), any()]),
          capture = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: all_of([base("!builtin.integer"), any()])]

  defop enumerable_reduce_range(
          start = all_of([base("!builtin.integer"), any()]),
          stop = all_of([base("!builtin.integer"), any()]),
          acc = all_of([base("!builtin.integer"), any()]),
          continuation = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: all_of([base("!builtin.integer"), any()])]

  defop enumerable_reduce_fun(
          enumerable = base(dyn()),
          acc = all_of([base("!builtin.integer"), any()]),
          reducer = any()
        ),
        results: [result: all_of([base("!builtin.integer"), any()])]

  defop func_addr(),
    attributes: [sym_name: base("#builtin.string")],
    results: [result: any()]

  defop list_head(list = base(dyn())),
    results: [result: base(dyn())]

  defop list_tail(list = base(dyn())),
    results: [result: base(dyn())]

  defop list_get(list = base(dyn()), index = all_of([base("!builtin.integer"), any()])),
    results: [result: base(dyn())]

  defop list_length(list = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop term_eq(left = base(dyn()), right = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop binary_length(binary = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop binary_get(binary = base(dyn()), index = all_of([base("!builtin.integer"), any()])),
    results: [result: base(dyn())]

  defop binary_slice(binary = base(dyn()), start = all_of([base("!builtin.integer"), any()])),
    results: [result: base(dyn())]

  defop binary_utf8_get(binary = base(dyn()), index = all_of([base("!builtin.integer"), any()])),
    results: [result: base(dyn())]

  defop binary_utf8_width(
          binary = base(dyn()),
          index = all_of([base("!builtin.integer"), any()])
        ),
        results: [result: all_of([base("!builtin.integer"), any()])]

  defop binary_utf8_length(binary = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop binary_encode16(binary = base(dyn())),
    results: [result: base(dyn())]

  defop binary_decode16(binary = base(dyn())),
    results: [result: base(dyn())]

  defop int_to_string(word = base(dyn())),
    results: [result: base(dyn())]

  defop string_to_int(binary = base(dyn())),
    results: [result: all_of([base("!builtin.integer"), any()])]

  defop yield(values = variadic(any())), traits: [:terminator]

  defop func(),
    attributes: [sym_name: base("#builtin.string")],
    regions: [body: {:region, size: 1}],
    traits: [:isolated_from_above]

  defop return(value = optional(any())), traits: [:terminator]
end
