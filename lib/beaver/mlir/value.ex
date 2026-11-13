alias Beaver.Native

defmodule Beaver.MLIR.Value do
  @moduledoc """
  MLIR SSA values: block arguments or operation results. Conditional use replacement traverses uses on a worker and
  dispatches predicates to the calling BEAM process through `Kinda.CallbackRuntime`.
  """
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR
  alias Kinda.CallbackRuntime
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Native
  def argument?(%__MODULE__{} = value), do: CAPI.mlirValueIsABlockArgument(value) |> Native.to_term()
  @doc "Returns whether the value is an operation result."
  def result?(%__MODULE__{} = value), do: CAPI.mlirValueIsAOpResult(value) |> Native.to_term()
  @doc "Returns the defining operation, or an error for a block argument."
  def owner(%__MODULE__{} = value),
    do: if(result?(value), do: {:ok, CAPI.mlirOpResultGetOwner(value)}, else: {:error, "not a result"})

  @doc "Returns the defining operation; raises for a block argument."
  def owner!(value) do
    case owner(value) do
      {:ok, op} -> op
      {:error, msg} -> raise ArgumentError, msg
    end
  end

  @doc "Returns the value's type."
  defdelegate type(value), to: CAPI, as: :mlirValueGetType
  @doc "Returns the source location carried by this value."
  defdelegate location(value), to: CAPI, as: :mlirValueGetLocation

  @doc """
  Sets a block argument's location and returns the value. Operation results inherit their operation's location and
  cannot be set independently.
  """
  @spec set_location(t(), MLIR.Location.source()) :: t()
  def set_location(%__MODULE__{} = value, loc) do
    unless argument?(value), do: raise(ArgumentError, "only block argument locations can be set")
    loc = MLIR.Location.resolve(loc, MLIR.context(value))
    :ok = CAPI.mlirBlockArgumentSetLocation(value, loc)
    value
  end

  @doc """
  Replaces uses whose `MLIR.OpOperand` predicate returns `true` on the calling BEAM process. Native traversal waits
  outside the scheduler through Kinda's callback runtime. Both values must share a context with thread-safe locking.
  Exceptions are re-raised with their stacktrace after traversal stops; replacements already accepted remain
  applied.
  """
  @spec replace_uses_with_if(t(), t(), (MLIR.OpOperand.t() -> boolean())) :: :ok
  def replace_uses_with_if(%__MODULE__{} = from, %__MODULE__{ref: replacement}, predicate)
      when is_function(predicate, 1) do
    id =
      CAPI.beaver_raw_execution_worker(MLIR.context(from).ref, :replace_uses, {from.ref, replacement, predicate}, false)

    await_conditional_replacement(id, nil)
  end

  defp await_conditional_replacement(id, exception) do
    receive do
      {:filter, token, {^id, predicate}, _dispatcher, op_operand_ref} ->
        outcome =
          CallbackRuntime.invoke(
            token,
            fn -> invoke_replace_predicate(predicate, op_operand_ref, exception) end,
            &CAPI.beaver_raw_callback_reply/2
          )

        await_conditional_replacement(id, exception || callback_exception(outcome))

      {^id, {{:ok, :ok}, nil}} ->
        finish_conditional_replacement(exception)

      {^id, {{:error, reason}, nil}} ->
        finish_conditional_replacement(
          exception || {:exception, :error, RuntimeError.exception(Atom.to_string(reason)), []}
        )
    end
  end

  defp invoke_replace_predicate(_predicate, _op_operand_ref, exception)
       when not is_nil(exception), do: {:error, :aborted}

  defp invoke_replace_predicate(predicate, op_operand_ref, nil) do
    op_operand = Native.check!(op_operand_ref)

    case predicate.(op_operand) do
      true -> {:ok, :replace}
      false -> {:error, :keep}
      other -> raise ArgumentError, "replacement predicate must return a boolean, got: #{inspect(other)}"
    end
  end

  defp callback_exception({:exception, _kind, _reason, _stacktrace} = exception), do: exception
  defp callback_exception(_outcome), do: nil
  defp finish_conditional_replacement(nil), do: :ok

  defp finish_conditional_replacement({:exception, kind, reason, stacktrace}),
    do: :erlang.raise(kind, reason, stacktrace)
end
