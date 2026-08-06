defmodule Beaver.MLIR.MemoryEffects do
  @moduledoc """
  Implements MLIR's `MemoryEffectsOpInterface` for dynamic operations.

  A callback returns `:pure` or a list of effect specifications. An effect is
  one of `:allocate`, `:free`, `:read`, or `:write`, optionally associated with
  an operand, result, block argument, or symbol:

      [
        {:read, {:operand, 0}},
        {:write, {:result, 0}},
        {:read, {:symbol, symbol_ref}, stage: 1}
      ]

  A bare effect such as `:write` is not associated with a particular IR
  entity. Options are `:parameters`, `:stage`, `:effect_on_full_region`, and
  `:resource`. The default resource and a null parameters attribute are used
  when omitted.

  The effects list passed to a two-argument callback is borrowed and valid only
  until that callback returns. Prefer returning declarative specifications.
  Callback exceptions are diagnosed and native code adds a conservative
  unknown write effect.
  """

  alias Beaver.MLIR

  @type effect() :: :allocate | :free | :read | :write
  @type target() ::
          nil
          | :operation
          | {:operand, non_neg_integer() | MLIR.OpOperand.t()}
          | {:result, non_neg_integer() | MLIR.Value.t()}
          | {:block_argument, MLIR.Value.t()}
          | {:symbol, MLIR.Attribute.t()}
          | MLIR.OpOperand.t()
          | MLIR.Value.t()
          | MLIR.Attribute.t()
  @type effect_spec() :: effect() | {effect(), target()} | {effect(), target(), keyword()}

  @callback memory_effects(MLIR.Operation.t()) ::
              :pure | [effect_spec()] | {:ok, [effect_spec()]}

  @doc "Attaches a callback-backed memory effects interface model."
  @spec attach(MLIR.Context.t(), String.t(), module() | function(), keyword()) ::
          MLIR.ExternalInterface.Attachment.t()
  def attach(context, operation_name, implementation, opts \\ []) do
    callback = callback!(implementation)

    MLIR.ExternalInterface.attach(
      context,
      operation_name,
      :memory_effects,
      %{get_effects: callback},
      opts
    )
  end

  @doc false
  def callback!(implementation) when is_atom(implementation) do
    unless function_exported?(implementation, :memory_effects, 1) do
      raise ArgumentError,
            "#{inspect(implementation)} must implement memory_effects/1"
    end

    &implementation.memory_effects/1
  end

  def callback!(callback) when is_function(callback, 1) or is_function(callback, 2),
    do: callback

  def callback!(other),
    do: raise(ArgumentError, "invalid memory effects implementation: #{inspect(other)}")

  @doc false
  @spec append(MLIR.MemoryEffectInstancesList.t(), MLIR.Operation.t(), [effect_spec()]) :: :ok
  def append(%MLIR.MemoryEffectInstancesList{} = effects, %MLIR.Operation{} = operation, specs)
      when is_list(specs) do
    Enum.each(specs, &append_one(effects, operation, &1))
    :ok
  end

  @doc "Adds the standard transform-dialect effects for read-only handle operands."
  def only_reads_handle(effects, operands),
    do: transform_handle_effect(:mlirTransformOnlyReadsHandle, effects, operands)

  @doc "Adds the standard transform-dialect effects for consumed handle operands."
  def consumes_handle(effects, operands),
    do: transform_handle_effect(:mlirTransformConsumesHandle, effects, operands)

  @doc "Adds the standard transform-dialect effects for produced handle results."
  def produces_handle(%MLIR.MemoryEffectInstancesList{} = effects, results) do
    results = List.wrap(results)
    validate_kind_list!(results, MLIR.Value, "transform results")
    array = Beaver.Native.array(results, MLIR.Value, mut: true)
    MLIR.CAPI.mlirTransformProducesHandle(array, length(results), effects)
    :ok
  end

  @doc "Marks a transform operation as potentially modifying payload IR."
  def modifies_payload(%MLIR.MemoryEffectInstancesList{} = effects) do
    MLIR.CAPI.mlirTransformModifiesPayload(effects)
    :ok
  end

  @doc "Marks a transform operation as only reading payload IR."
  def only_reads_payload(%MLIR.MemoryEffectInstancesList{} = effects) do
    MLIR.CAPI.mlirTransformOnlyReadsPayload(effects)
    :ok
  end

  defp transform_handle_effect(function, %MLIR.MemoryEffectInstancesList{} = effects, operands) do
    operands = List.wrap(operands)
    validate_kind_list!(operands, MLIR.OpOperand, "transform operands")
    array = Beaver.Native.array(operands, MLIR.OpOperand, mut: true)
    apply(MLIR.CAPI, function, [array, length(operands), effects])
    :ok
  end

  defp append_one(effects, operation, effect) when effect in [:allocate, :free, :read, :write],
    do: append_one(effects, operation, {effect, nil, []})

  defp append_one(effects, operation, {effect, target})
       when effect in [:allocate, :free, :read, :write],
       do: append_one(effects, operation, {effect, target, []})

  defp append_one(effects, operation, {effect, target, opts})
       when effect in [:allocate, :free, :read, :write] and is_list(opts) do
    effect = effect_handle(effect)
    parameters = Keyword.get(opts, :parameters, MLIR.Attribute.null())
    stage = Keyword.get(opts, :stage, 0)
    full_region = Keyword.get(opts, :effect_on_full_region, false)
    resource = Keyword.get_lazy(opts, :resource, &MLIR.CAPI.mlirSideEffectsDefaultResourceGet/0)

    unless is_struct(parameters, MLIR.Attribute) do
      raise ArgumentError, ":parameters must be an MLIR attribute"
    end

    unless is_integer(stage) do
      raise ArgumentError, ":stage must be an integer"
    end

    unless is_boolean(full_region) do
      raise ArgumentError, ":effect_on_full_region must be a boolean"
    end

    unless is_struct(resource, MLIR.SideEffectResource) do
      raise ArgumentError, ":resource must be an MLIR side effect resource"
    end

    instance =
      create_instance(
        effect,
        normalize_target(operation, target),
        parameters,
        stage,
        full_region,
        resource
      )

    try do
      MLIR.CAPI.mlirMemoryEffectInstancesListAppend(effects, instance)
    after
      MLIR.CAPI.mlirMemoryEffectInstanceDestroy(instance)
    end
  end

  defp append_one(_effects, _operation, other) do
    raise ArgumentError, "invalid memory effect specification: #{inspect(other)}"
  end

  defp effect_handle(:allocate), do: MLIR.CAPI.mlirMemoryEffectsAllocateGet()
  defp effect_handle(:free), do: MLIR.CAPI.mlirMemoryEffectsFreeGet()
  defp effect_handle(:read), do: MLIR.CAPI.mlirMemoryEffectsReadGet()
  defp effect_handle(:write), do: MLIR.CAPI.mlirMemoryEffectsWriteGet()

  defp normalize_target(_operation, target) when target in [nil, :operation], do: nil

  defp normalize_target(operation, {:operand, index}) when is_integer(index) and index >= 0,
    do: MLIR.CAPI.mlirOperationGetOpOperand(operation, index)

  defp normalize_target(_operation, {:operand, %MLIR.OpOperand{} = operand}), do: operand

  defp normalize_target(operation, {:result, index}) when is_integer(index) and index >= 0,
    do: MLIR.Operation.result(operation, index)

  defp normalize_target(_operation, {:result, %MLIR.Value{} = result}), do: result

  defp normalize_target(_operation, {:block_argument, %MLIR.Value{} = argument}),
    do: argument

  defp normalize_target(_operation, {:symbol, %MLIR.Attribute{} = symbol}), do: symbol
  defp normalize_target(_operation, %MLIR.OpOperand{} = operand), do: operand
  defp normalize_target(_operation, %MLIR.Value{} = value), do: value
  defp normalize_target(_operation, %MLIR.Attribute{} = symbol), do: symbol

  defp normalize_target(_operation, target),
    do: raise(ArgumentError, "invalid memory effect target: #{inspect(target)}")

  defp create_instance(effect, nil, parameters, stage, full_region, resource),
    do: MLIR.CAPI.mlirMemoryEffectInstanceCreate(effect, parameters, stage, full_region, resource)

  defp create_instance(
         effect,
         %MLIR.OpOperand{} = operand,
         parameters,
         stage,
         full_region,
         resource
       ),
       do:
         MLIR.CAPI.mlirMemoryEffectInstanceCreateForOpOperand(
           effect,
           operand,
           parameters,
           stage,
           full_region,
           resource
         )

  defp create_instance(effect, %MLIR.Value{} = value, parameters, stage, full_region, resource) do
    cond do
      MLIR.Value.result?(value) ->
        MLIR.CAPI.mlirMemoryEffectInstanceCreateForOpResult(
          effect,
          value,
          parameters,
          stage,
          full_region,
          resource
        )

      MLIR.Value.argument?(value) ->
        MLIR.CAPI.mlirMemoryEffectInstanceCreateForBlockArgument(
          effect,
          value,
          parameters,
          stage,
          full_region,
          resource
        )

      true ->
        raise ArgumentError,
              "memory effect value target is neither an op result nor block argument"
    end
  end

  defp create_instance(
         effect,
         %MLIR.Attribute{} = symbol,
         parameters,
         stage,
         full_region,
         resource
       ),
       do:
         MLIR.CAPI.mlirMemoryEffectInstanceCreateForSymbol(
           effect,
           symbol,
           parameters,
           stage,
           full_region,
           resource
         )

  defp validate_kind_list!(values, module, label) do
    unless Enum.all?(values, &is_struct(&1, module)) do
      raise ArgumentError, "#{label} must contain only #{inspect(module)} values"
    end
  end
end
