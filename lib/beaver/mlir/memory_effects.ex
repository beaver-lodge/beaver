alias Beaver.MLIR.{Attribute, CAPI, OpOperand, Operation, Value}
alias Beaver.Native

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

  Bare effects have no target. Options are `:parameters` (default null),
  `:stage`, `:effect_on_full_region`, and `:resource` (default resource).
  A two-argument callback's effects list is borrowed until it returns.
  Prefer declarative specifications; callback exceptions emit diagnostics and
  add a conservative unknown write effect.
  """

  alias Beaver.MLIR

  @type effect() :: :allocate | :free | :read | :write
  @type target() ::
          nil
          | :operation
          | {:operand, non_neg_integer() | OpOperand.t()}
          | {:result, non_neg_integer() | Value.t()}
          | {:block_argument, Value.t()}
          | {:symbol, Attribute.t()}
          | OpOperand.t()
          | Value.t()
          | Attribute.t()
  @type effect_spec() :: effect() | {effect(), target()} | {effect(), target(), keyword()}

  @callback memory_effects(Operation.t()) ::
              :pure | [effect_spec()] | {:ok, [effect_spec()]}

  @doc "Attaches a callback-backed memory effects interface model."
  @spec attach(MLIR.Context.t(), String.t(), module() | function(), keyword()) ::
          MLIR.ExternalInterface.Attachment.t()
  def attach(ctx, op_name, implementation, opts \\ []) do
    callback = callback!(implementation)

    MLIR.ExternalInterface.attach(ctx, op_name, :memory_effects, %{get_effects: callback}, opts)
  end

  @doc false
  def callback!(implementation) when is_atom(implementation) do
    unless function_exported?(implementation, :memory_effects, 1),
      do: raise(ArgumentError, "#{inspect(implementation)} must implement memory_effects/1")

    &implementation.memory_effects/1
  end

  def callback!(callback) when is_function(callback, 1) or is_function(callback, 2),
    do: callback

  def callback!(other),
    do: raise(ArgumentError, "invalid memory effects implementation: #{inspect(other)}")

  @doc """
  Returns declarative effects, each including `:parameters`, `:stage`,
  `:effect_on_full_region` and `:resource`, without retaining callback-scoped handles.
  """
  @spec get(Operation.t(), keyword()) :: [effect_spec()]
  def get(%Operation{} = op, opts \\ []) do
    timeout_ms = Beaver.Execution.callback_timeout!(opts)

    case Beaver.Execution.run(MLIR.context(op), :memory_effects, {op.ref}, timeout: timeout_ms + 1_000) do
      :error -> raise ArgumentError, "operation does not implement MemoryEffectsOpInterface"
      instances -> Enum.map(instances, &decode_instance/1)
    end
  end

  @doc false
  @spec append(MLIR.MemoryEffectInstancesList.t(), Operation.t(), [effect_spec()]) :: :ok
  def append(%MLIR.MemoryEffectInstancesList{} = effects, %Operation{} = op, specs)
      when is_list(specs) do
    Enum.each(specs, &append_one(effects, op, &1))
    :ok
  end

  for {name, function, kind, label, doc} <- [
        {:only_reads_handle, :beaverTransformOnlyReadsHandle, OpOperand, "transform operands",
         "Adds read-only handle effects."},
        {:consumes_handle, :beaverTransformConsumesHandle, OpOperand, "transform operands",
         "Adds consumed handle effects."},
        {:produces_handle, :beaverTransformProducesHandle, Value, "transform results", "Adds produced handle effects."}
      ] do
    @doc doc
    def unquote(name)(%MLIR.MemoryEffectInstancesList{} = effects, values) do
      values = List.wrap(values)
      validate_kind_list!(values, unquote(kind), unquote(label))
      CAPI.unquote(function)(Native.array(values, unquote(kind), mut: true), length(values), effects)
      :ok
    end
  end

  @doc "Marks a transform operation as potentially modifying payload IR."
  def modifies_payload(%MLIR.MemoryEffectInstancesList{} = effects) do
    CAPI.beaverTransformModifiesPayload(effects)
    :ok
  end

  @doc "Marks a transform operation as only reading payload IR."
  def only_reads_payload(%MLIR.MemoryEffectInstancesList{} = effects) do
    CAPI.beaverTransformOnlyReadsPayload(effects)
    :ok
  end

  defp append_one(effects, op, effect) when effect in [:allocate, :free, :read, :write],
    do: append_one(effects, op, {effect, nil, []})

  defp append_one(effects, op, {effect, target})
       when effect in [:allocate, :free, :read, :write],
       do: append_one(effects, op, {effect, target, []})

  defp append_one(effects, op, {effect, target, opts})
       when effect in [:allocate, :free, :read, :write] and is_list(opts) do
    effect = effect_handle(effect)
    parameters = Keyword.get(opts, :parameters, Attribute.null())
    stage = Keyword.get(opts, :stage, 0)
    full_region = Keyword.get(opts, :effect_on_full_region, false)
    resource = Keyword.get_lazy(opts, :resource, &CAPI.mlirSideEffectsDefaultResourceGet/0)

    unless is_struct(parameters, Attribute),
      do: raise(ArgumentError, ":parameters must be an MLIR attribute")

    unless is_integer(stage), do: raise(ArgumentError, ":stage must be an integer")

    unless is_boolean(full_region),
      do: raise(ArgumentError, ":effect_on_full_region must be a boolean")

    unless is_struct(resource, MLIR.SideEffectResource),
      do: raise(ArgumentError, ":resource must be an MLIR side effect resource")

    instance =
      create_instance(effect, normalize_target(op, target), parameters, stage, full_region, resource)

    try do
      CAPI.beaverMemoryEffectInstancesListAppend(effects, instance)
    after
      CAPI.mlirMemoryEffectInstanceDestroy(instance)
    end
  end

  defp append_one(_effects, _operation, other),
    do: raise(ArgumentError, "invalid memory effect specification: #{inspect(other)}")

  defp effect_handle(:allocate), do: CAPI.mlirMemoryEffectsAllocateGet()
  defp effect_handle(:free), do: CAPI.mlirMemoryEffectsFreeGet()
  defp effect_handle(:read), do: CAPI.mlirMemoryEffectsReadGet()
  defp effect_handle(:write), do: CAPI.mlirMemoryEffectsWriteGet()

  defp decode_instance({kind, target_kind, target, parameters, stage, full_region, resource}) do
    {decode_effect(kind), decode_target(target_kind, Native.normalize(target)),
     parameters: Native.normalize(parameters),
     stage: stage,
     effect_on_full_region: full_region,
     resource: Native.normalize(resource)}
  end

  defp decode_effect(0), do: :allocate
  defp decode_effect(1), do: :free
  defp decode_effect(2), do: :read
  defp decode_effect(3), do: :write

  defp decode_effect(other),
    do: raise(ArgumentError, "unsupported native memory effect kind: #{inspect(other)}")

  defp decode_target(:none, nil), do: nil
  defp decode_target(:operand, %OpOperand{} = operand), do: {:operand, operand}

  defp decode_target(:value, %Value{} = value),
    do: if(Value.result?(value), do: {:result, value}, else: {:block_argument, value})

  defp decode_target(:symbol, %Attribute{} = symbol), do: {:symbol, symbol}

  defp normalize_target(_operation, target) when target in [nil, :operation], do: nil

  defp normalize_target(op, {:operand, index}) when is_integer(index) and index >= 0,
    do: CAPI.mlirOperationGetOpOperand(op, index)

  defp normalize_target(op, {:result, index}) when is_integer(index) and index >= 0,
    do: Operation.result(op, index)

  defp normalize_target(_operation, {tag, %kind{} = value})
       when {tag, kind} in [{:operand, OpOperand}, {:result, Value}, {:block_argument, Value}, {:symbol, Attribute}],
       do: value

  defp normalize_target(_operation, %kind{} = value) when kind in [OpOperand, Value, Attribute], do: value

  defp normalize_target(_operation, target),
    do: raise(ArgumentError, "invalid memory effect target: #{inspect(target)}")

  defp create_instance(effect, nil, parameters, stage, full_region, resource),
    do: CAPI.mlirMemoryEffectInstanceCreate(effect, parameters, stage, full_region, resource)

  defp create_instance(effect, %OpOperand{} = operand, parameters, stage, full_region, resource),
    do: CAPI.mlirMemoryEffectInstanceCreateForOpOperand(effect, operand, parameters, stage, full_region, resource)

  defp create_instance(effect, %Value{} = value, parameters, stage, full_region, resource) do
    cond do
      Value.result?(value) ->
        CAPI.mlirMemoryEffectInstanceCreateForOpResult(effect, value, parameters, stage, full_region, resource)

      Value.argument?(value) ->
        CAPI.mlirMemoryEffectInstanceCreateForBlockArgument(effect, value, parameters, stage, full_region, resource)

      true ->
        raise ArgumentError,
              "memory effect value target is neither an op result nor block argument"
    end
  end

  defp create_instance(effect, %Attribute{} = symbol, parameters, stage, full_region, resource),
    do: CAPI.mlirMemoryEffectInstanceCreateForSymbol(effect, symbol, parameters, stage, full_region, resource)

  defp validate_kind_list!(values, module, label) do
    unless Enum.all?(values, &is_struct(&1, module)),
      do: raise(ArgumentError, "#{label} must contain only #{inspect(module)} values")
  end
end
