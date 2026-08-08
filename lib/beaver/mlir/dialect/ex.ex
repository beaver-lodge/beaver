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
  # homogeneous, which would reject mixed-typed calls.
  defop call(
          arg0 = optional(any()),
          arg1 = optional(any()),
          arg2 = optional(any()),
          arg3 = optional(any())
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

  defop tuple(elements = variadic(base(dyn()))),
    results: [result: base(dyn())]

  defop list(elements = variadic(base(dyn()))),
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

  defop list_head(list = base(dyn())),
    results: [result: base(dyn())]

  defop list_tail(list = base(dyn())),
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

  defop yield(values = variadic(any())), traits: [:terminator]

  defop func(),
    attributes: [sym_name: base("#builtin.string")],
    regions: [body: {:region, size: 1}],
    traits: [:isolated_from_above]

  defop return(value = optional(any())), traits: [:terminator]
end
