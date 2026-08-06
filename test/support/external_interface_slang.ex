defmodule ExternalInterfaceSlang do
  @moduledoc false
  use Beaver.Slang, name: "external_interface_test"

  def pure_effects(_operation), do: :pure
  def write_effects(_operation), do: [:write]
  def speculatability(_operation), do: :speculatable

  def transform_apply(_operation, _rewriter, _results, _state), do: :ok
  def repeated_handles?(_operation), do: true
  def fail_apply(_operation, _rewriter, _results, _state), do: raise("expected callback failure")

  def forward_effects(operation, effects) do
    operand = Beaver.MLIR.CAPI.mlirOperationGetOpOperand(operation, 0)
    result = Beaver.MLIR.Operation.result(operation, 0)
    Beaver.MLIR.MemoryEffects.only_reads_handle(effects, operand)
    Beaver.MLIR.MemoryEffects.produces_handle(effects, result)
    Beaver.MLIR.MemoryEffects.only_reads_payload(effects)
  end

  def mark_effects(operation, effects) do
    operand = Beaver.MLIR.CAPI.mlirOperationGetOpOperand(operation, 0)
    Beaver.MLIR.MemoryEffects.only_reads_handle(effects, operand)
    Beaver.MLIR.MemoryEffects.modifies_payload(effects)
  end

  def forward_apply(operation, _rewriter, _results, state) do
    handle = Beaver.MLIR.CAPI.mlirOperationGetOperand(operation, 0)

    payload =
      state
      |> Beaver.MLIR.TransformOpInterface.payload_ops(handle)
      |> Enum.flat_map(&payload_functions/1)

    {:ok, %{0 => {:ops, payload}}}
  end

  defp payload_functions(root) do
    {_, matches} = Beaver.Walker.prewalk(root, [], &collect_payload_function/2)
    Enum.reverse(matches)
  end

  defp collect_payload_function(%Beaver.MLIR.Operation{} = operation, acc) do
    acc =
      if Beaver.MLIR.Operation.name(operation) == "func.func", do: [operation | acc], else: acc

    {operation, acc}
  end

  defp collect_payload_function(element, acc), do: {element, acc}

  def mark_apply(operation, rewriter, _results, state) do
    handle = Beaver.MLIR.CAPI.mlirOperationGetOperand(operation, 0)
    base = Beaver.MLIR.TransformOpInterface.rewriter_base(rewriter)

    state
    |> Beaver.MLIR.TransformOpInterface.payload_ops(handle)
    |> Enum.each(fn payload ->
      Beaver.MLIR.RewriterBase.start_op_modification(base, payload)

      Beaver.MLIR.CAPI.mlirOperationSetAttributeByName(
        payload,
        Beaver.MLIR.StringRef.create("external_interface_test.marked"),
        Beaver.MLIR.Attribute.unit(ctx: Beaver.MLIR.context(payload))
      )

      Beaver.MLIR.RewriterBase.finalize_op_modification(base, payload)
    end)

    :ok
  end

  def populate_patterns(operation, patterns) do
    Beaver.MLIR.RewritePatternSet.add(
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
      base = Beaver.MLIR.PatternRewriter.as_base(rewriter)
      Beaver.MLIR.RewriterBase.start_op_modification(base, operation)

      Beaver.MLIR.CAPI.mlirOperationSetAttributeByName(
        operation,
        Beaver.MLIR.StringRef.create("external_interface_test.pattern_applied"),
        Beaver.MLIR.Attribute.unit(ctx: Beaver.MLIR.context(operation))
      )

      Beaver.MLIR.RewriterBase.finalize_op_modification(base, operation)
      {:ok, state}
    end
  end

  defop pure(),
    interfaces: [
      memory_effects: &__MODULE__.pure_effects/1,
      conditionally_speculatable: &__MODULE__.speculatability/1
    ]

  defop write(), interfaces: [memory_effects: &__MODULE__.write_effects/1]

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
    interfaces: [
      memory_effects: &__MODULE__.forward_effects/2,
      transform_op: [apply: &__MODULE__.forward_apply/4]
    ]

  defop mark(handle = base("!transform.any_op")),
    interfaces: [
      memory_effects: &__MODULE__.mark_effects/2,
      transform_op: [apply: &__MODULE__.mark_apply/4]
    ]

  defop fail(handle = base("!transform.any_op")),
    interfaces: [
      memory_effects: &__MODULE__.mark_effects/2,
      transform_op: [apply: &__MODULE__.fail_apply/4]
    ]

  defop patterns(),
    interfaces: [pattern_descriptor: [populate_patterns: &__MODULE__.populate_patterns/2]]
end
