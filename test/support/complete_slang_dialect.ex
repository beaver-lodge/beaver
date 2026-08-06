defmodule CompleteSlang do
  @moduledoc false
  use Beaver.Slang, name: "complete_slang"

  defconstraint integer_like do
    all_of([base("!builtin.integer"), any()])
  end

  defconstraint direction_value do
    any_of([
      Beaver.MLIR.Attribute.string("left"),
      Beaver.MLIR.Attribute.string("right")
    ])
  end

  deftype box(element = ^integer_like)
  deftype token()
  deftype named_parameters(any()), parameter_names: [:item]
  defattr direction(value = ^direction_value)

  defop(consume(value = any()))
  defop(consume_token(value = base(token())))

  defop sequence(head = any(), tail = variadic(any()), fallback = optional(any())),
    results: [value: optional(any())]

  defop named_io(single(any())),
    operand_names: [:input],
    results: [single(any())],
    result_names: [:output]

  defop scope(),
    attributes: [label: base("#builtin.string")],
    regions: [body: {:region, args: [], size: 1}],
    traits: [:isolated_from_above, :no_terminator]

  defop yield(), traits: [:terminator]
end
