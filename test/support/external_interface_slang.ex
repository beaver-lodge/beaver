defmodule ExternalInterfaceSlang do
  @moduledoc false
  alias Beaver.MLIR.{Attribute, CAPI, MemoryEffects, Operation, PatternRewriter, RewriterBase, RewritePatternSet}
  alias Beaver.MLIR.{TransformOpInterface}

  use Beaver.Slang, name: "external_interface_test"

  def pure_effects(_operation), do: :pure
  def write_effects(_operation), do: [:write]
  def speculatability(_operation), do: :speculatable

  def transform_apply(_operation, _rewriter, _results, _state), do: :ok
  def repeated_handles?(_operation), do: true
  def fail_apply(_operation, _rewriter, _results, _state), do: raise("expected callback failure")

  def forward_effects(operation, effects) do
    operand = CAPI.mlirOperationGetOpOperand(operation, 0)
    result = Operation.result(operation, 0)
    MemoryEffects.only_reads_handle(effects, operand)
    MemoryEffects.produces_handle(effects, result)
    MemoryEffects.only_reads_payload(effects)
  end

  def mark_effects(operation, effects) do
    operand = CAPI.mlirOperationGetOpOperand(operation, 0)
    MemoryEffects.only_reads_handle(effects, operand)
    MemoryEffects.modifies_payload(effects)
  end

  def forward_apply(operation, _rewriter, _results, state) do
    handle = CAPI.mlirOperationGetOperand(operation, 0)

    payload = state |> TransformOpInterface.payload_ops(handle) |> Enum.flat_map(&payload_functions/1)

    {:ok, %{0 => {:ops, payload}}}
  end

  defp payload_functions(root), do: Beaver.Walker.find_ops(root, "func.func")

  def mark_apply(operation, rewriter, _results, state) do
    handle = CAPI.mlirOperationGetOperand(operation, 0)
    base = TransformOpInterface.rewriter_base(rewriter)

    state
    |> TransformOpInterface.payload_ops(handle)
    |> Enum.each(&mark(base, &1, "marked"))

    :ok
  end

  def populate_patterns(operation, patterns) do
    RewritePatternSet.add(
      patterns,
      "func.return",
      &__MODULE__.mark_return/4,
      ctx: Beaver.MLIR.context(operation)
    )

    :ok
  end

  def mark_return(_pattern, operation, rewriter, state) do
    if operation["external_interface_test.pattern_applied"] do
      {:error, state}
    else
      base = PatternRewriter.as_base(rewriter)
      mark(base, operation, "pattern_applied")
      {:ok, state}
    end
  end

  defp mark(base, operation, name) do
    RewriterBase.start_op_modification(base, operation)

    Operation.put_discardable_attribute(
      operation,
      "external_interface_test." <> name,
      Attribute.unit(ctx: Beaver.MLIR.context(operation))
    )

    RewriterBase.finalize_op_modification(base, operation)
  end

  defop pure(),
    interfaces: [
      memory_effects: &__MODULE__.pure_effects/1,
      conditionally_speculatable: &__MODULE__.speculatability/1
    ]

  defop write(), interfaces: [memory_effects: &__MODULE__.write_effects/1]

  defop detached()

  defop transform(),
    interfaces: [
      transform_op: [
        apply: &__MODULE__.transform_apply/4,
        allows_repeated_handle_operands: &__MODULE__.repeated_handles?/1
      ],
      pattern_descriptor: [populate_patterns: &__MODULE__.populate_patterns/2]
    ]

  defop forward(handle = base("!transform.any_op")),
    do: [base("!transform.any_op")],
    interfaces: [memory_effects: &__MODULE__.forward_effects/2, transform_op: [apply: &__MODULE__.forward_apply/4]]

  defop mark(handle = base("!transform.any_op")),
    interfaces: [memory_effects: &__MODULE__.mark_effects/2, transform_op: [apply: &__MODULE__.mark_apply/4]]

  defop fail(handle = base("!transform.any_op")),
    interfaces: [memory_effects: &__MODULE__.mark_effects/2, transform_op: [apply: &__MODULE__.fail_apply/4]]

  defop patterns(), interfaces: [pattern_descriptor: [populate_patterns: &__MODULE__.populate_patterns/2]]
end
