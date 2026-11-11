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

  @callbacks [
    {:conversion, :add_conversion, quote(do: (Type.t() -> conversion_result())), 1},
    {:one_to_n, :add_1_to_n_conversion, quote(do: (Type.t() -> one_to_n_result())), 1},
    {:source_materialization, :add_source_materialization,
     quote(do: (RewriterBase.t(), Type.t(), [Value.t()], Location.t() -> materialization_result())), 4},
    {:target_materialization, :add_target_materialization,
     quote(do: (RewriterBase.t(), Type.t(), [Value.t()], Location.t(), Type.t() -> materialization_result())), 5},
    {:one_to_n_target_materialization, :add_1_to_n_target_materialization,
     quote(
       do: (RewriterBase.t(), [Type.t()], [Value.t()], Location.t(), Type.t() | nil ->
              one_to_n_materialization_result())
     ), 5}
  ]

  @callback_options for {key, add, _, _} <- @callbacks, do: {key, add}

  @spec create(keyword()) :: t()
  def create(opts \\ []) when is_list(opts) do
    allowed = [:timeout | Keyword.keys(@callback_options)]

    case Keyword.keys(opts) -- allowed do
      [] -> :ok
      unsupported -> raise ArgumentError, "unsupported TypeConverter options: #{inspect(unsupported)}"
    end

    timeout_ms = Beaver.Execution.callback_timeout!(opts)

    {:managed_type_converter, converter_term, registration} = CAPI.beaver_raw_type_converter_create()

    %__MODULE__{ref: ref} = Native.check!(converter_term)
    converter = %__MODULE__{ref: ref, registration: registration, timeout_ms: timeout_ms}

    for {key, add} <- @callback_options,
        callback <- Enum.flat_map(Keyword.get_values(opts, key), &if(is_list(&1), do: &1, else: [&1])),
        do: apply(__MODULE__, add, [converter, callback])

    converter
  end

  for {_key, name, callback_type, arity} <- @callbacks do
    @spec unquote(name)(t(), unquote(callback_type)) :: t()
    def unquote(name)(%__MODULE__{} = converter, callback) when is_function(callback, unquote(arity)) do
      :ok =
        apply(CAPI, unquote(:"beaver_raw_type_converter_#{name}"), [
          converter.registration,
          callback,
          converter.timeout_ms
        ])

      converter
    end
  end

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
    fallback = Keyword.get(opts, :fallback, :identity)
    recursive = Keyword.get(opts, :recursive, false)

    unless Keyword.keys(opts) -- [:fallback, :recursive] == [] and fallback in [:identity, :declined] and
             is_boolean(recursive),
           do:
             raise(
               ArgumentError,
               "conversion map options must be :fallback (:identity | :declined) and :recursive (boolean)"
             )

    identity_fallback = fallback == :identity
    mode_arg = if recursive, do: {identity_fallback, true}, else: identity_fallback

    :ok =
      CAPI.beaver_raw_type_converter_add_conversion_map(
        converter.registration,
        Enum.map(sources, & &1.ref),
        target.ref,
        mode_arg
      )

    converter
  end

  def add_conversion_map(%__MODULE__{}, sources, %Type{}, _opts) when is_list(sources),
    do: raise(ArgumentError, "conversion map requires at least one source type")

  @spec convert(t(), Type.t()) :: {:ok, Type.t()} | {:error, term()}
  def convert(%__MODULE__{} = converter, %Type{} = type) do
    id =
      CAPI.beaver_raw_execution_worker(MLIR.context(type).ref, :convert_type, {converter.registration, type.ref}, false)

    await_conversion(id, converter.timeout_ms, nil)
  end

  @doc "Converts `type` with `converter`, returning the converted `Type.t()` or raising `ArgumentError`."
  @spec convert!(t(), Type.t()) :: Type.t()
  def convert!(%__MODULE__{} = converter, %Type{} = type) do
    case convert(converter, type) do
      {:ok, converted} ->
        converted

      {:error, reason} ->
        raise ArgumentError,
              "TypeConverter failed to convert #{MLIR.to_string(type)}: #{inspect(reason)}"
    end
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
