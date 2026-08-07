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

  defop call(args = variadic(any())),
    results: [result: base(dyn())],
    attributes: [callee: ^callee_value, arity: ^integer_value]

  defop func(),
    attributes: [sym_name: base("#builtin.string")],
    regions: [body: {:region, size: 1}],
    traits: [:isolated_from_above]

  defop return(value = optional(any())), traits: [:terminator]
end
