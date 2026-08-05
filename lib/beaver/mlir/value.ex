defmodule Beaver.MLIR.Value do
  @moduledoc """
  This module handles MLIR values, which represent SSA (Static Single Assignment) values in the IR.

  Values can be either block arguments or operation results. That's why this module provides
  functions to check if a value is an argument or a result (`argument?/1`, `result?/1`), or to get the owner of a result (`owner/1`).

  Conditional use replacement runs its native traversal on an MLIR worker
  thread and dispatches each use to the calling BEAM process through
  `Kinda.CallbackRuntime`.
  """
  alias Beaver.MLIR.CAPI
  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  def argument?(%__MODULE__{} = value) do
    CAPI.mlirValueIsABlockArgument(value) |> Beaver.Native.to_term()
  end

  @doc """
  Returns true if the value is a result of an operation.
  """
  def result?(%__MODULE__{} = value) do
    CAPI.mlirValueIsAOpResult(value) |> Beaver.Native.to_term()
  end

  @doc """
  Return the defining op of this value if this value is a result
  """
  def owner(%__MODULE__{} = value) do
    if result?(value) do
      {:ok, CAPI.mlirOpResultGetOwner(value)}
    else
      {:error, "not a result"}
    end
  end

  @doc """
  Return the defining op of this value. Raises if this value is not a result
  """
  def owner!(value) do
    case owner(value) do
      {:ok, op} ->
        op

      {:error, msg} ->
        raise ArgumentError, msg
    end
  end

  @doc """
  Return the type of this value
  """
  defdelegate type(value), to: CAPI, as: :mlirValueGetType

  @doc """
  Replaces uses of `value` for which `predicate` returns `true`.

  The predicate receives an `#{inspect(MLIR.OpOperand)}` and executes in the
  calling BEAM process. Native traversal runs outside the scheduler and waits
  through Kinda's resource-backed callback runtime. The source and replacement
  values must belong to the same multithreaded MLIR context.

  Predicate exceptions are re-raised with their original stacktrace after the
  native traversal stops. This operation is not transactional: replacements
  accepted before an exception remain applied.
  """
  @spec replace_uses_with_if(t(), t(), (MLIR.OpOperand.t() -> boolean())) :: :ok
  def replace_uses_with_if(
        %__MODULE__{ref: from},
        %__MODULE__{ref: replacement},
        predicate
      )
      when is_function(predicate, 1) do
    case CAPI.beaver_raw_value_replace_uses_with_if(from, replacement, predicate) do
      {:async, id} -> await_conditional_replacement(id, nil)
      other -> Beaver.Native.check!(other)
    end
  end

  defp await_conditional_replacement(id, exception) do
    receive do
      {:filter, token, predicate, ^id, op_operand_ref} ->
        outcome =
          CallbackRuntime.invoke(
            token,
            fn -> invoke_replace_predicate(predicate, op_operand_ref, exception) end,
            &CAPI.beaver_raw_callback_reply/2
          )

        await_conditional_replacement(id, exception || callback_exception(outcome))

      {:replace_uses_with_if_done, ^id} ->
        finish_conditional_replacement(exception)

      {:replace_uses_with_if_error, ^id, reason} ->
        finish_conditional_replacement(
          exception || {:exception, :error, RuntimeError.exception(Atom.to_string(reason)), []}
        )
    end
  end

  defp invoke_replace_predicate(_predicate, _op_operand_ref, exception)
       when not is_nil(exception),
       do: {:error, :aborted}

  defp invoke_replace_predicate(predicate, op_operand_ref, nil) do
    op_operand = Beaver.Native.check!(op_operand_ref)

    case predicate.(op_operand) do
      true ->
        {:ok, :replace}

      false ->
        {:error, :keep}

      other ->
        raise ArgumentError, "replacement predicate must return a boolean, got: #{inspect(other)}"
    end
  end

  defp callback_exception({:exception, _kind, _reason, _stacktrace} = exception), do: exception
  defp callback_exception(_outcome), do: nil

  defp finish_conditional_replacement(nil), do: :ok

  defp finish_conditional_replacement({:exception, kind, reason, stacktrace}) do
    :erlang.raise(kind, reason, stacktrace)
  end
end
