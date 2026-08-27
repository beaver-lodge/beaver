defmodule Beaver.MLIR.TypeConverter do
  @moduledoc """
  An owning MLIR type converter whose Elixir callbacks are safe across native
  worker threads, process termination, and NIF upgrades.

  Conversion callbacks may return one type (1:1), multiple types (1:N), an
  empty list (type erasure), `:declined`, or `{:error, reason}`.
  """

  use Kinda.ResourceKind,
    raw_module: Beaver.MLIR.CAPI.Raw,
    codec: Beaver.Native,
    fields: [registration: nil, timeout_ms: 30_000]

  alias Beaver.MLIR

  @type conversion_result() ::
          MLIR.Type.t() | {:ok, MLIR.Type.t()} | :declined | {:error, term()}
  @type one_to_n_result() ::
          [MLIR.Type.t()] | {:ok, [MLIR.Type.t()]} | :declined | {:error, term()}
  @type materialization_result() ::
          MLIR.Value.t() | {:ok, MLIR.Value.t()} | :declined | {:error, term()}
  @type one_to_n_materialization_result() ::
          [MLIR.Value.t()] | {:ok, [MLIR.Value.t()]} | :declined | {:error, term()}

  @spec create(keyword()) :: t()
  def create(opts \\ []) when is_list(opts) do
    allowed = [
      :timeout,
      :conversion,
      :one_to_n,
      :source_materialization,
      :target_materialization,
      :one_to_n_target_materialization
    ]

    case Keyword.keys(opts) -- allowed do
      [] ->
        :ok

      unsupported ->
        raise ArgumentError, "unsupported TypeConverter options: #{inspect(unsupported)}"
    end

    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    unless is_integer(timeout_ms) and timeout_ms >= 0 do
      raise ArgumentError, ":timeout must be a non-negative integer"
    end

    {:managed_type_converter, converter_term, registration} =
      MLIR.CAPI.beaver_raw_type_converter_create()

    %__MODULE__{ref: ref} = Beaver.Native.check!(converter_term)
    converter = %__MODULE__{ref: ref, registration: registration, timeout_ms: timeout_ms}

    opts
    |> callback_values(:conversion)
    |> Enum.each(&add_conversion(converter, &1))

    opts
    |> callback_values(:one_to_n)
    |> Enum.each(&add_1_to_n_conversion(converter, &1))

    opts
    |> callback_values(:source_materialization)
    |> Enum.each(&add_source_materialization(converter, &1))

    opts
    |> callback_values(:target_materialization)
    |> Enum.each(&add_target_materialization(converter, &1))

    opts
    |> callback_values(:one_to_n_target_materialization)
    |> Enum.each(&add_1_to_n_target_materialization(converter, &1))

    converter
  end

  defp callback_values(opts, key) do
    opts
    |> Keyword.get_values(key)
    |> Enum.flat_map(fn
      callbacks when is_list(callbacks) -> callbacks
      callback -> [callback]
    end)
  end

  @spec add_conversion(t(), (MLIR.Type.t() -> conversion_result())) :: t()
  def add_conversion(%__MODULE__{} = converter, callback) when is_function(callback, 1) do
    :ok =
      MLIR.CAPI.beaver_raw_type_converter_add_conversion(
        converter.registration,
        callback,
        converter.timeout_ms
      )

    converter
  end

  @doc """
  Adds a native, callback-free mapping from any source type to one target type.

  All types must belong to the same context used by the conversion. The map is
  stored by the native converter and does not invoke the BEAM while MLIR asks
  repeated type-conversion questions. `:fallback` is `:identity` by default;
  use `:declined` to let a later converter callback handle unmatched types.
  """
  @spec add_conversion_map(t(), [MLIR.Type.t()], MLIR.Type.t(), keyword()) :: t()
  def add_conversion_map(converter, sources, target, opts \\ [])

  def add_conversion_map(
        %__MODULE__{} = converter,
        [%MLIR.Type{} | _] = sources,
        %MLIR.Type{} = target,
        opts
      ) do
    identity_fallback = Keyword.get(opts, :fallback, :identity) == :identity

    unless Keyword.keys(opts) -- [:fallback] == [] and
             Keyword.get(opts, :fallback, :identity) in [:identity, :declined] do
      raise ArgumentError, "conversion map :fallback must be :identity or :declined"
    end

    :ok =
      MLIR.CAPI.beaver_raw_type_converter_add_conversion_map(
        converter.registration,
        Enum.map(sources, & &1.ref),
        target.ref,
        identity_fallback
      )

    converter
  end

  def add_conversion_map(%__MODULE__{}, sources, %MLIR.Type{}, _opts) when is_list(sources) do
    raise ArgumentError, "conversion map requires at least one source type"
  end

  @spec add_1_to_n_conversion(t(), (MLIR.Type.t() -> one_to_n_result())) :: t()
  def add_1_to_n_conversion(%__MODULE__{} = converter, callback) when is_function(callback, 1) do
    :ok =
      MLIR.CAPI.beaver_raw_type_converter_add_1_to_n_conversion(
        converter.registration,
        callback,
        converter.timeout_ms
      )

    converter
  end

  @spec add_source_materialization(
          t(),
          (MLIR.RewriterBase.t(), MLIR.Type.t(), [MLIR.Value.t()], MLIR.Location.t() ->
             materialization_result())
        ) :: t()
  def add_source_materialization(%__MODULE__{} = converter, callback)
      when is_function(callback, 4) do
    :ok =
      MLIR.CAPI.beaver_raw_type_converter_add_source_materialization(
        converter.registration,
        callback,
        converter.timeout_ms
      )

    converter
  end

  @spec add_target_materialization(
          t(),
          (MLIR.RewriterBase.t(),
           MLIR.Type.t(),
           [MLIR.Value.t()],
           MLIR.Location.t(),
           MLIR.Type.t() ->
             materialization_result())
        ) :: t()
  def add_target_materialization(%__MODULE__{} = converter, callback)
      when is_function(callback, 5) do
    :ok =
      MLIR.CAPI.beaver_raw_type_converter_add_target_materialization(
        converter.registration,
        callback,
        converter.timeout_ms
      )

    converter
  end

  @spec add_1_to_n_target_materialization(
          t(),
          (MLIR.RewriterBase.t(),
           [MLIR.Type.t()],
           [MLIR.Value.t()],
           MLIR.Location.t(),
           MLIR.Type.t()
           | nil ->
             one_to_n_materialization_result())
        ) :: t()
  def add_1_to_n_target_materialization(%__MODULE__{} = converter, callback)
      when is_function(callback, 5) do
    :ok =
      MLIR.CAPI.beaver_raw_type_converter_add_1_to_n_target_materialization(
        converter.registration,
        callback,
        converter.timeout_ms
      )

    converter
  end

  @spec convert(t(), MLIR.Type.t()) :: {:ok, MLIR.Type.t()} | {:error, term()}
  def convert(%__MODULE__{} = converter, %MLIR.Type{ref: type_ref}) do
    id = MLIR.CAPI.beaver_raw_type_converter_convert_async(converter.registration, type_ref)
    await_conversion(id, converter.timeout_ms, nil)
  end

  defp await_conversion(id, timeout_ms, callback_failure) do
    receive do
      {:type_converter_done, ^id, result} ->
        finish_conversion(result, callback_failure)

      {:convert_type, _token, _callback, _callback_id, _sent_at, _type} = message ->
        {:handled, failure} = MLIR.Conversion.Callbacks.handle(message)
        await_conversion(id, timeout_ms, callback_failure || failure)

      {:convert_types, _token, _callback, _callback_id, _sent_at, _type} = message ->
        {:handled, failure} = MLIR.Conversion.Callbacks.handle(message)
        await_conversion(id, timeout_ms, callback_failure || failure)
    after
      timeout_ms + 1_000 ->
        {:error, :timeout}
    end
  end

  defp finish_conversion(_result, {:exception, kind, reason, stacktrace}) do
    :erlang.raise(kind, reason, stacktrace)
  end

  defp finish_conversion(_result, {:error, reason}), do: {:error, reason}
  defp finish_conversion(nil, nil), do: {:error, :conversion_failed}
  defp finish_conversion(result, nil), do: {:ok, Beaver.Native.check!(result)}

  @spec destroy(t()) :: :ok
  def destroy(%__MODULE__{registration: registration}) do
    MLIR.CAPI.beaver_raw_type_converter_destroy(registration)
  end

  @spec with(keyword(), (t() -> result)) :: result when result: var
  def with(opts \\ [], fun) when is_function(fun, 1) do
    converter = create(opts)

    try do
      fun.(converter)
    after
      destroy(converter)
    end
  end
end
