defmodule Beaver.MLIR.TypeConverter.Callback do
  @moduledoc """
  A callback-backed MLIR type converter.

  Conversion runs on a native worker thread. The process which creates the
  converter owns its callback mailbox and projects successful `%MLIR.Type{}`
  results back into the native callback before replying.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  alias Kinda.CallbackRuntime

  defstruct [:converter, :registration, :callback, timeout_ms: 30_000]

  @type callback_result ::
          MLIR.Type.t()
          | {:ok, MLIR.Type.t()}
          | :declined
          | {:error, term()}
  @type t :: %__MODULE__{
          converter: MLIR.TypeConverter.t(),
          registration: reference(),
          callback: (MLIR.Type.t() -> callback_result()),
          timeout_ms: non_neg_integer()
        }

  @spec create((MLIR.Type.t() -> callback_result()), keyword()) :: t()
  def create(callback, opts \\ []) when is_function(callback, 1) do
    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    {:callback_type_converter, converter_term, registration} =
      CAPI.beaver_raw_type_converter_create_callback(callback, timeout_ms)

    converter = Beaver.Native.check!(converter_term)

    %__MODULE__{
      converter: converter,
      registration: registration,
      callback: callback,
      timeout_ms: timeout_ms
    }
  end

  @spec convert(t(), MLIR.Type.t()) :: {:ok, MLIR.Type.t()} | {:error, term()}
  def convert(%__MODULE__{} = converter, %MLIR.Type{ref: type_ref}) do
    id =
      CAPI.beaver_raw_type_converter_convert_async(converter.registration, type_ref)

    await_conversion(converter, id, nil)
  end

  @spec destroy(t()) :: :ok
  def destroy(%__MODULE__{} = converter) do
    id = CAPI.beaver_raw_type_converter_destroy_async(converter.registration)

    receive do
      {:type_converter_destroyed, ^id} -> :ok
    after
      converter.timeout_ms + 1_000 -> raise "timed out destroying callback-backed TypeConverter"
    end
  end

  defp await_conversion(converter, id, callback_failure) do
    receive do
      {:convert_type, token, callback, ^id, type_term} ->
        outcome =
          CallbackRuntime.invoke_reply(
            token,
            fn -> invoke_callback(callback, Beaver.Native.check!(type_term)) end,
            &reply_conversion/2
          )

        await_conversion(converter, id, callback_failure || callback_failure(outcome))

      {:type_converter_done, ^id, result} ->
        finish_conversion(result, callback_failure)

      {:type_converter_error, reason} ->
        {:error, reason}
    after
      converter.timeout_ms + 1_000 ->
        {:error, :timeout}
    end
  end

  defp invoke_callback(callback, type) do
    case callback.(type) do
      %MLIR.Type{} = converted -> {:ok, {:success, converted}}
      {:ok, %MLIR.Type{} = converted} -> {:ok, {:success, converted}}
      :declined -> {:ok, :declined}
      {:error, _reason} = error -> error
      other -> raise ArgumentError, "invalid TypeConverter callback result: #{inspect(other)}"
    end
  end

  defp reply_conversion(token, {:ok, {:success, %MLIR.Type{ref: converted_ref}}}) do
    CAPI.beaver_raw_type_converter_reply_callback(token, true, 0, converted_ref)
  end

  defp reply_conversion(token, {:ok, :declined}) do
    CAPI.beaver_raw_type_converter_reply_callback(token, true, 2, nil)
  end

  defp reply_conversion(token, {:error, _reason}) do
    CAPI.beaver_raw_type_converter_reply_callback(token, true, 1, nil)
  end

  defp reply_conversion(token, {:exception, _kind, _reason, _stacktrace}) do
    CAPI.beaver_raw_type_converter_reply_callback(token, false, 1, nil)
  end

  defp callback_failure({:error, reason}), do: {:error, reason}

  defp callback_failure({:exception, _kind, _reason, _stacktrace} = exception),
    do: exception

  defp callback_failure(_outcome), do: nil

  defp finish_conversion(_result, {:exception, kind, reason, stacktrace}) do
    :erlang.raise(kind, reason, stacktrace)
  end

  defp finish_conversion(_result, {:error, reason}), do: {:error, reason}
  defp finish_conversion(nil, nil), do: {:error, :conversion_failed}
  defp finish_conversion(result, nil), do: {:ok, Beaver.Native.check!(result)}
end
