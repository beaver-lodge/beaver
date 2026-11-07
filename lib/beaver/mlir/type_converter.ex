alias Beaver.MLIR.{CAPI, Location, RewriterBase, Type, Value}
alias Beaver.Native

defmodule Beaver.MLIR.TypeConverter do
  @moduledoc """
  An owning MLIR type converter whose Elixir callbacks are safe across native
  worker threads, process termination, and NIF upgrades.

  Conversion callbacks may return one type (1:1), multiple types (1:N), an
  empty list (type erasure), `:declined`, or `{:error, reason}`.
  """

  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native, fields: [registration: nil, timeout_ms: 30_000]

  alias Beaver.MLIR

  @type conversion_result() ::
          Type.t() | {:ok, Type.t()} | :declined | {:error, term()}
  @type one_to_n_result() ::
          [Type.t()] | {:ok, [Type.t()]} | :declined | {:error, term()}
  @type materialization_result() ::
          Value.t() | {:ok, Value.t()} | :declined | {:error, term()}
  @type one_to_n_materialization_result() ::
          [Value.t()] | {:ok, [Value.t()]} | :declined | {:error, term()}

  @spec create(keyword()) :: t()
  def create(opts \\ []) when is_list(opts) do
    allowed = ~w(
      timeout conversion one_to_n source_materialization target_materialization one_to_n_target_materialization
    )a

    case Keyword.keys(opts) -- allowed do
      [] -> :ok
      unsupported -> raise ArgumentError, "unsupported TypeConverter options: #{inspect(unsupported)}"
    end

    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    unless is_integer(timeout_ms) and timeout_ms >= 0,
      do: raise(ArgumentError, ":timeout must be a non-negative integer")

    {:managed_type_converter, converter_term, registration} = CAPI.beaver_raw_type_converter_create()

    %__MODULE__{ref: ref} = Native.check!(converter_term)
    converter = %__MODULE__{ref: ref, registration: registration, timeout_ms: timeout_ms}

    for {key, add} <- [
          conversion: &add_conversion/2,
          one_to_n: &add_1_to_n_conversion/2,
          source_materialization: &add_source_materialization/2,
          target_materialization: &add_target_materialization/2,
          one_to_n_target_materialization: &add_1_to_n_target_materialization/2
        ],
        callback <- callback_values(opts, key),
        do: add.(converter, callback)

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

  @spec add_conversion(t(), (Type.t() -> conversion_result())) :: t()
  def add_conversion(%__MODULE__{} = converter, callback) when is_function(callback, 1),
    do: add_callback(converter, callback, :beaver_raw_type_converter_add_conversion)

  @doc """
  Adds a native, callback-free mapping from any source type to one target type.

  All types must belong to the same context used by the conversion. The map is
  stored by the native converter and does not invoke the BEAM while MLIR asks
  repeated type-conversion questions. `:fallback` is `:identity` by default;
  use `:declined` to let a later converter callback handle unmatched types.
  """
  @spec add_conversion_map(t(), [Type.t()], Type.t(), keyword()) :: t()
  def add_conversion_map(converter, sources, target, opts \\ [])

  def add_conversion_map(%__MODULE__{} = converter, [%Type{} | _] = sources, %Type{} = target, opts) do
    identity_fallback = Keyword.get(opts, :fallback, :identity) == :identity

    unless Keyword.keys(opts) -- [:fallback] == [] and Keyword.get(opts, :fallback, :identity) in [:identity, :declined],
      do: raise(ArgumentError, "conversion map :fallback must be :identity or :declined")

    :ok =
      CAPI.beaver_raw_type_converter_add_conversion_map(
        converter.registration,
        Enum.map(sources, & &1.ref),
        target.ref,
        identity_fallback
      )

    converter
  end

  def add_conversion_map(%__MODULE__{}, sources, %Type{}, _opts) when is_list(sources),
    do: raise(ArgumentError, "conversion map requires at least one source type")

  @spec add_1_to_n_conversion(t(), (Type.t() -> one_to_n_result())) :: t()
  def add_1_to_n_conversion(%__MODULE__{} = converter, callback) when is_function(callback, 1),
    do: add_callback(converter, callback, :beaver_raw_type_converter_add_1_to_n_conversion)

  @spec add_source_materialization(t(), (RewriterBase.t(), Type.t(), [Value.t()], Location.t() ->
                                           materialization_result())) :: t()
  def add_source_materialization(%__MODULE__{} = converter, callback)
      when is_function(callback, 4),
      do: add_callback(converter, callback, :beaver_raw_type_converter_add_source_materialization)

  @spec add_target_materialization(t(), (RewriterBase.t(), Type.t(), [Value.t()], Location.t(), Type.t() ->
                                           materialization_result())) :: t()
  def add_target_materialization(%__MODULE__{} = converter, callback)
      when is_function(callback, 5),
      do: add_callback(converter, callback, :beaver_raw_type_converter_add_target_materialization)

  @spec add_1_to_n_target_materialization(t(), (RewriterBase.t(),
                                                [Type.t()],
                                                [Value.t()],
                                                Location.t(),
                                                Type.t()
                                                | nil ->
                                                  one_to_n_materialization_result())) :: t()
  def add_1_to_n_target_materialization(%__MODULE__{} = converter, callback)
      when is_function(callback, 5),
      do: add_callback(converter, callback, :beaver_raw_type_converter_add_1_to_n_target_materialization)

  defp add_callback(converter, callback, function) do
    :ok = apply(CAPI, function, [converter.registration, callback, converter.timeout_ms])
    converter
  end

  @spec convert(t(), Type.t()) :: {:ok, Type.t()} | {:error, term()}
  def convert(%__MODULE__{} = converter, %Type{} = type) do
    id =
      CAPI.beaver_raw_execution_worker(MLIR.context(type).ref, :convert_type, {converter.registration, type.ref}, false)

    await_conversion(id, converter.timeout_ms, nil)
  end

  defp await_conversion(id, timeout_ms, callback_failure) do
    receive do
      {^id, {{:ok, result}, nil}} ->
        finish_conversion(result, callback_failure)

      {^id, {{:error, reason}, nil}} ->
        {:error, reason}

      {kind, _token, _callback, _callback_id, _sent_at, _type} = message
      when kind in [:convert_type, :convert_types] ->
        {:handled, failure} = MLIR.Conversion.Callbacks.handle(message)
        await_conversion(id, timeout_ms, callback_failure || failure)
    after
      timeout_ms + 1_000 -> {:error, :timeout}
    end
  end

  defp finish_conversion(_result, {:exception, kind, reason, stacktrace}), do: :erlang.raise(kind, reason, stacktrace)

  defp finish_conversion(_result, {:error, reason}), do: {:error, reason}
  defp finish_conversion(nil, nil), do: {:error, :conversion_failed}
  defp finish_conversion(result, nil), do: {:ok, Native.check!(result)}

  @spec destroy(t()) :: :ok
  def destroy(%__MODULE__{registration: registration}),
    do: CAPI.beaver_raw_type_converter_destroy(registration)

  @spec with(keyword(), (t() -> result)) :: result when result: var
  def with(opts \\ [], fun) when is_function(fun, 1),
    do: Beaver.Native.with_resource(create(opts), &destroy/1, fun)
end
