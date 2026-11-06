alias Beaver.MLIR.{CAPI, ConversionTarget, Operation, RewritePatternSet, Type, Value}

defmodule Beaver.MLIR.Conversion do
  @moduledoc """
  High-level, diagnostic-aware MLIR dialect conversion.

  For a declarative, inspectable, and reusable pipeline, see `Beaver.MLIR.Conversion.Plan`.

  `apply/5` runs full or partial conversion outside BEAM scheduler threads and
  services callback-backed targets, type converters, and conversion patterns
  in the calling process. Mutable pattern sets are frozen and cleaned up
  automatically. See the [dialect conversion guide](dialect-conversion.html)
  for a complete Slang-to-LLVM example.
  """

  use Beaver.ComposerGenerator, prefix: "mlirCreateConversion"

  require Logger
  alias Beaver.MLIR

  defmodule Error do
    @moduledoc "An error returned by a failed dialect conversion."
    defexception [:mode, diagnostics: [], callback_failure: nil, reason: :conversion_failed]

    @type t() :: %__MODULE__{
            mode: Beaver.MLIR.Conversion.mode(),
            diagnostics: [term()],
            callback_failure: term(),
            reason: term()
          }

    @impl true
    def message(%__MODULE__{} = error) do
      prefix = "#{error.mode} dialect conversion failed"

      details =
        case error.callback_failure do
          nil ->
            prefix

          {:error, reason} ->
            "#{prefix}; callback failed: #{inspect(reason)}"

          {:exception, kind, reason, stacktrace} ->
            "#{prefix}; callback raised:\n" <> Exception.format(kind, reason, stacktrace)
        end

      if error.diagnostics == [],
        do: details,
        else: MLIR.Diagnostic.format(error.diagnostics, details)
    end
  end

  @type mode() :: :full | :partial
  @type conversion_ir() :: MLIR.Module.t() | Operation.t()
  @type result() :: {:ok, conversion_ir(), [term()]} | {:error, Error.t()}

  @spec apply(
          mode(),
          conversion_ir(),
          ConversionTarget.t(),
          RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: result()
  def apply(mode, ir, target, patterns, opts \\ [])

  def apply(mode, ir, target, patterns, opts), do: run(mode, ir, target, patterns, opts, false)

  @doc """
  Runs one conversion with bounded callback and native-worker profiling.

  Returns the ordinary conversion result together with a machine-readable
  `Beaver.MLIR.Conversion.Profile` receipt. Profiling is opt-in so normal
  conversion keeps its existing callback path and measurement overhead.
  """
  @spec profile(
          mode(),
          conversion_ir(),
          ConversionTarget.t(),
          RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: {result(), Beaver.MLIR.Conversion.Profile.receipt()}
  def profile(mode, ir, target, patterns, opts \\ []),
    do: run(mode, ir, target, patterns, opts, true)

  defp run(mode, ir, target, patterns, opts, profile?)

  defp run(mode, ir, %ConversionTarget{} = target, %RewritePatternSet{} = set, opts, profile?)
       when mode in [:full, :partial] do
    conversion_options = conversion_options!(opts)
    frozen = RewritePatternSet.freeze(set)

    do_apply(mode, ir, target, frozen, true, conversion_options, profile?)
  end

  defp run(mode, ir, %ConversionTarget{} = target, %MLIR.FrozenRewritePatternSet{} = patterns, opts, profile?)
       when mode in [:full, :partial] do
    do_apply(mode, ir, target, patterns, false, conversion_options!(opts), profile?)
  end

  defp do_apply(
         mode,
         ir,
         %ConversionTarget{registration: registration},
         %MLIR.FrozenRewritePatternSet{ref: patterns_ref},
         owns_patterns,
         {folding_mode, materializations, timeout_ms},
         profile?
       ) do
    operation = operation(ir)
    context = MLIR.context(ir)
    profile = if profile?, do: __MODULE__.Profile.start(mode, ir)

    id =
      try do
        CAPI.beaver_raw_apply_conversion_async(
          mode == :full,
          registration,
          operation.ref,
          patterns_ref,
          owns_patterns,
          folding_mode,
          materializations,
          profile?
        )
      rescue
        exception ->
          if owns_patterns,
            do: MLIR.FrozenRewritePatternSet.threaded_destroy(context, %MLIR.FrozenRewritePatternSet{ref: patterns_ref})

          reraise exception, __STACKTRACE__
      end

    await(id, mode, ir, timeout_ms, nil, profile)
  end

  defp conversion_options!(opts) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "conversion options must be a keyword list")

    unsupported = Keyword.keys(opts) -- [:timeout, :folding_mode, :build_materializations]

    if unsupported != [], do: raise(ArgumentError, "unsupported conversion options: #{inspect(unsupported)}")

    folding_mode = folding_mode!(Keyword.get(opts, :folding_mode))
    materializations = materializations!(Keyword.get(opts, :build_materializations))

    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    unless is_integer(timeout_ms) and timeout_ms >= 0,
      do: raise(ArgumentError, ":timeout must be a non-negative integer")

    {folding_mode, materializations, timeout_ms}
  end

  defp folding_mode!(nil), do: -1
  defp folding_mode!(:never), do: 0
  defp folding_mode!(:before_patterns), do: 1
  defp folding_mode!(:after_patterns), do: 2

  defp folding_mode!(mode), do: raise(ArgumentError, "unsupported conversion folding mode: #{inspect(mode)}")

  defp materializations!(nil), do: -1
  defp materializations!(false), do: 0
  defp materializations!(true), do: 1

  defp materializations!(value),
    do: raise(ArgumentError, "build_materializations must be boolean, got: #{inspect(value)}")

  @spec full(conversion_ir(), ConversionTarget.t(), RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(), keyword()) ::
          result()
  def full(ir, target, patterns, opts \\ []), do: apply(:full, ir, target, patterns, opts)

  @spec partial(
          conversion_ir(),
          ConversionTarget.t(),
          RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: result()
  def partial(ir, target, patterns, opts \\ []), do: apply(:partial, ir, target, patterns, opts)

  @spec apply!(
          mode(),
          conversion_ir(),
          ConversionTarget.t(),
          RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: conversion_ir()
  def apply!(mode, ir, target, patterns, opts \\ []) do
    case apply(mode, ir, target, patterns, opts) do
      {:ok, converted, _diagnostics} -> converted
      {:error, error} -> raise error
    end
  end

  defp operation(%Operation{} = op), do: op
  defp operation(%MLIR.Module{} = module), do: Operation.from_module(module)

  defp await(id, mode, ir, timeout_ms, callback_failure, profile) do
    receive do
      {:conversion_done, ^id, result, native_profile} ->
        conversion_result = finish(mode, ir, result, callback_failure)

        if profile,
          do: {conversion_result, __MODULE__.Profile.finish(profile, conversion_result, ir, native_profile)},
          else: conversion_result

      {name, _token, _callback, _callback_id, _sent_at, _arg} = message
      when name in [:conversion_legality, :convert_type, :convert_types] ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure, profile)

      {name, _token, _callback, _callback_id, _sent_at, _arg1, _arg2, _arg3} = message
      when name in [:conversion_pattern, :conversion_pattern_1_to_n] ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure, profile)

      {:source_materialization, _token, _callback, _callback_id, _sent_at, _rewriter, _type, _inputs, _loc} = message ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure, profile)

      {:target_materialization, _token, _callback, _callback_id, _sent_at, _rewriter, _type, _inputs, _loc,
       _original_type} = message ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure, profile)

      {:target_materialization_1_to_n, _token, _callback, _callback_id, _sent_at, _rewriter, _output_types, _inputs,
       _loc, _original_type} = message ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure, profile)
    after
      timeout_ms + 1_000 ->
        Logger.warning("still waiting for #{mode} dialect conversion to finish")
        await(id, mode, ir, timeout_ms, callback_failure, profile)
    end
  end

  defp handle_and_await(message, id, mode, ir, timeout_ms, callback_failure, profile) do
    started_at = System.monotonic_time(:nanosecond)
    {:handled, failure} = __MODULE__.Callbacks.handle(message)
    stopped_at = System.monotonic_time(:nanosecond)

    profile =
      if profile do
        service_ns = max(stopped_at - started_at, 0)
        native_wait_ns = max(System.os_time(:nanosecond) - elem(message, 4), service_ns)

        __MODULE__.Profile.record_callback(
          profile,
          elem(message, 0),
          callback_root(message),
          service_ns,
          native_wait_ns
        )
      end

    await(id, mode, ir, timeout_ms, callback_failure || failure, profile)
  end

  defp callback_root(message) do
    case elem(message, 0) do
      kind when kind in [:conversion_pattern, :conversion_pattern_1_to_n] ->
        {root_name, _callback} = MLIR.ConversionPattern.unwrap_callback(elem(message, 2))
        root_name

      _kind ->
        nil
    end
  end

  defp finish(mode, _ir, nil, callback_failure),
    do: {:error, %Error{mode: mode, callback_failure: callback_failure, reason: :native_worker_failed}}

  defp finish(mode, ir, result, callback_failure) do
    {logical_result, diagnostics} = Beaver.Native.check!(result)

    if MLIR.LogicalResult.success?(logical_result) and is_nil(callback_failure),
      do: {:ok, ir, diagnostics},
      else: {:error, %Error{mode: mode, diagnostics: diagnostics, callback_failure: callback_failure}}
  end
end

defmodule Beaver.MLIR.Conversion.Callbacks do
  @moduledoc false

  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  # Keep the exact native message shapes while sharing argument decoding and
  # exception capture. Pattern registrations additionally carry a root name.
  for {kind, arity, normalize, reply} <- [
        {:conversion_legality, 1, :legality, :reply_legality},
        {:convert_type, 1, :converted_type, :reply_type},
        {:convert_types, 1, :converted_types, :reply_types},
        {:source_materialization, 4, :materialized, :reply_value},
        {:target_materialization, 5, :materialized, :reply_value},
        {:target_materialization_1_to_n, 5, :materialized_values, :reply_values},
        {:conversion_pattern, 3, :pattern_result, :reply_pattern},
        {:conversion_pattern_1_to_n, 3, :pattern_result, :reply_pattern}
      ] do
    args = Macro.generate_arguments(arity, __MODULE__)

    def handle({unquote(kind), token, callback, _id, _sent_at, unquote_splicing(args)}) do
      callback =
        if unquote(kind) in [:conversion_pattern, :conversion_pattern_1_to_n],
          do: elem(MLIR.ConversionPattern.unwrap_callback(callback), 1),
          else: callback

      invoke(
        token,
        fn -> unquote(normalize)(apply(callback, Enum.map([unquote_splicing(args)], &native/1))) end,
        &unquote(reply)(&1, &2)
      )
    end
  end

  def handle(_message), do: :unhandled

  defp invoke(token, fun, reply) do
    outcome = CallbackRuntime.invoke_reply(token, fun, reply)
    {:handled, callback_failure(outcome)}
  end

  defp native(nil), do: nil
  defp native(values) when is_list(values), do: Enum.map(values, &native/1)
  defp native(value), do: Beaver.Native.check!(value)

  defp legality(:legal), do: {:ok, :legal}
  defp legality(true), do: {:ok, :legal}
  defp legality(:illegal), do: {:ok, :illegal}
  defp legality(false), do: {:ok, :illegal}
  defp legality(:no_opinion), do: {:ok, :no_opinion}
  defp legality({:error, _reason} = error), do: error

  defp legality(other), do: raise(ArgumentError, "invalid dynamic legality result: #{inspect(other)}")

  defp converted_type(%Type{} = type), do: {:ok, {:success, type}}
  defp converted_type({:ok, %Type{} = type}), do: {:ok, {:success, type}}
  defp converted_type(:declined), do: {:ok, :declined}
  defp converted_type({:error, _reason} = error), do: error

  defp converted_type(other), do: raise(ArgumentError, "invalid type conversion result: #{inspect(other)}")

  defp converted_types(types) when is_list(types) do
    validate_list!(types, Type, "1:N type conversion")
    {:ok, {:success, types}}
  end

  defp converted_types({:ok, types}) when is_list(types), do: converted_types(types)
  defp converted_types(:declined), do: {:ok, :declined}
  defp converted_types({:error, _reason} = error), do: error

  defp converted_types(other), do: raise(ArgumentError, "invalid 1:N conversion result: #{inspect(other)}")

  defp materialized(%Value{} = value), do: {:ok, {:success, value}}
  defp materialized({:ok, %Value{} = value}), do: {:ok, {:success, value}}
  defp materialized(nil), do: {:ok, :declined}
  defp materialized(:declined), do: {:ok, :declined}
  defp materialized({:error, _reason} = error), do: error

  defp materialized(other), do: raise(ArgumentError, "invalid materialization result: #{inspect(other)}")

  defp materialized_values(values) when is_list(values) do
    validate_list!(values, Value, "1:N target materialization")
    {:ok, {:success, values}}
  end

  defp materialized_values({:ok, values}) when is_list(values), do: materialized_values(values)
  defp materialized_values(:declined), do: {:ok, :declined}
  defp materialized_values({:error, _reason} = error), do: error

  defp materialized_values(other), do: raise(ArgumentError, "invalid 1:N materialization result: #{inspect(other)}")

  defp pattern_result(:ok), do: {:ok, :match}
  defp pattern_result(true), do: {:ok, :match}
  defp pattern_result({:ok, _state}), do: {:ok, :match}
  defp pattern_result(:no_match), do: {:ok, :no_match}
  defp pattern_result(false), do: {:ok, :no_match}
  defp pattern_result({:error, _reason} = error), do: error

  defp pattern_result(other), do: raise(ArgumentError, "invalid conversion pattern result: #{inspect(other)}")

  defp validate_list!(values, module, label) do
    unless Enum.all?(values, &is_struct(&1, module)),
      do: raise(ArgumentError, "#{label} must return only #{inspect(module)} values")
  end

  defp reply_legality(token, {:ok, result}) do
    code = %{legal: 0, illegal: 1, no_opinion: 2} |> Map.fetch!(result)
    CAPI.beaver_raw_callback_reply_code(token, true, code)
  end

  defp reply_legality(token, _failure), do: CAPI.beaver_raw_callback_reply_code(token, false, 1)

  defp reply_type(token, {:ok, {:success, %Type{ref: ref}}}),
    do: CAPI.beaver_raw_type_converter_reply_callback(token, true, 0, ref)

  defp reply_type(token, {:ok, :declined}), do: CAPI.beaver_raw_type_converter_reply_callback(token, true, 2, nil)

  defp reply_type(token, _failure), do: CAPI.beaver_raw_type_converter_reply_callback(token, false, 1, nil)

  defp reply_types(token, {:ok, {:success, types}}),
    do: CAPI.beaver_raw_type_converter_reply_types(token, true, 0, Enum.map(types, & &1.ref))

  defp reply_types(token, {:ok, :declined}), do: CAPI.beaver_raw_type_converter_reply_types(token, true, 2, [])

  defp reply_types(token, _failure), do: CAPI.beaver_raw_type_converter_reply_types(token, false, 1, [])

  defp reply_value(token, {:ok, {:success, %Value{ref: ref}}}),
    do: CAPI.beaver_raw_type_converter_reply_value(token, true, ref)

  defp reply_value(token, {:ok, :declined}), do: CAPI.beaver_raw_type_converter_reply_value(token, true, nil)

  defp reply_value(token, _failure), do: CAPI.beaver_raw_type_converter_reply_value(token, false, nil)

  defp reply_values(token, {:ok, {:success, values}}),
    do: CAPI.beaver_raw_type_converter_reply_values(token, true, 0, Enum.map(values, & &1.ref))

  defp reply_values(token, _failure), do: CAPI.beaver_raw_type_converter_reply_values(token, false, 1, [])

  defp reply_pattern(token, {:ok, :match}), do: CAPI.beaver_raw_callback_reply(token, true)

  defp reply_pattern(token, _no_match_or_failure), do: CAPI.beaver_raw_callback_reply(token, false)

  defp callback_failure({:error, reason}), do: {:error, reason}

  defp callback_failure({:exception, _kind, _reason, _stacktrace} = exception), do: exception

  defp callback_failure(_outcome), do: nil
end
