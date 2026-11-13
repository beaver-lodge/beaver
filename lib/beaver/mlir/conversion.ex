alias Beaver.MLIR.{CAPI, ConversionTarget, Operation, RewritePatternSet, Type, Value}

defmodule Beaver.MLIR.Conversion.Callbacks do
  @moduledoc false
  alias Beaver.MLIR
  alias Kinda.CallbackRuntime
  # Keep the exact native message shapes while sharing argument decoding and
  # exception capture. Pattern registrations additionally carry a root name.
  @callbacks [
    {:conversion_legality, 1, :legality, :reply_legality},
    {:convert_type, 1, :converted_type, :reply_type},
    {:convert_types, 1, :converted_types, :reply_types},
    {:source_materialization, 4, :materialized, :reply_value},
    {:target_materialization, 5, :materialized, :reply_value},
    {:target_materialization_1_to_n, 5, :materialized_values, :reply_values},
    {:conversion_pattern, 3, :pattern_result, :reply_pattern},
    {:conversion_pattern_1_to_n, 3, :pattern_result, :reply_pattern}
  ]
  @messages Enum.map(@callbacks, fn {kind, arity, _, _} -> {kind, arity + 5} end)
  defguard is_message(message) when is_tuple(message) and {elem(message, 0), tuple_size(message)} in @messages

  for {kind, arity, normalize, reply} <- @callbacks do
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
    {:handled, MLIR.ExternalInterface.callback_failure(outcome)}
  end

  defp native(nil), do: nil
  defp native(values) when is_list(values), do: Enum.map(values, &native/1)
  defp native(value), do: Beaver.Native.check!(value)
  defp legality(value) when value in [:legal, true], do: {:ok, :legal}
  defp legality(value) when value in [:illegal, false], do: {:ok, :illegal}
  defp legality(:no_opinion), do: {:ok, :no_opinion}
  defp legality({:error, _reason} = error), do: error
  defp legality(other), do: raise(ArgumentError, "invalid dynamic legality result: #{inspect(other)}")
  defguardp handle_result(value, module, many) when (many and is_list(value)) or (not many and is_struct(value, module))

  for {name, module, many, label, list_label} <- [
        {:converted_type, Type, false, "type conversion", nil},
        {:converted_types, Type, true, "1:N conversion", "1:N type conversion"},
        {:materialized, Value, false, "materialization", nil},
        {:materialized_values, Value, true, "1:N materialization", "1:N target materialization"}
      ] do
    defp unquote(name)({:ok, value}) when handle_result(value, unquote(module), unquote(many)), do: unquote(name)(value)

    defp unquote(name)(value) when handle_result(value, unquote(module), unquote(many)) do
      if unquote(many), do: validate_list!(value, unquote(module), unquote(list_label))
      {:ok, {:success, value}}
    end

    if name == :materialized do
      defp unquote(name)(nil), do: {:ok, :declined}
    end

    defp unquote(name)(:declined), do: {:ok, :declined}
    defp unquote(name)({:error, _} = error), do: error
    defp unquote(name)(other), do: raise(ArgumentError, "invalid #{unquote(label)} result: #{inspect(other)}")
  end

  defp pattern_result(value) when value in [:ok, true], do: {:ok, :match}
  defp pattern_result({:ok, _state}), do: {:ok, :match}
  defp pattern_result(value) when value in [:no_match, false], do: {:ok, :no_match}
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

  for {name, nif, kind} <- [
        {:reply_type, :beaver_raw_type_converter_reply_callback, :type},
        {:reply_types, :beaver_raw_type_converter_reply_types, :types},
        {:reply_value, :beaver_raw_type_converter_reply_value, :value},
        {:reply_values, :beaver_raw_type_converter_reply_values, :values}
      ] do
    defp unquote(name)(token, outcome) do
      {success, code, value} =
        case outcome do
          {:ok, {:success, value}} -> {true, 0, value}
          {:ok, :declined} when unquote(kind) != :values -> {true, 2, nil}
          _ -> {false, 1, nil}
        end

      value = if unquote(kind) in [:types, :values], do: Enum.map(value || [], & &1.ref), else: value && value.ref
      args = if unquote(kind) == :value, do: [token, success, value], else: [token, success, code, value]
      apply(CAPI, unquote(nif), args)
    end
  end

  defp reply_pattern(token, {:ok, :match}), do: CAPI.beaver_raw_callback_reply(token, true)
  defp reply_pattern(token, _no_match_or_failure), do: CAPI.beaver_raw_callback_reply(token, false)
end

defmodule Beaver.MLIR.Conversion do
  @moduledoc """
  High-level, diagnostic-aware MLIR dialect conversion.

  For a declarative, inspectable, and reusable pipeline, see `Beaver.MLIR.Conversion.Plan`.

  `apply/5` runs full or partial conversion outside BEAM scheduler threads and services callback-backed targets,
  type converters, and conversion patterns in the calling process. Mutable pattern sets are frozen and cleaned up
  automatically. See the [dialect conversion guide](dialect-conversion.html) for a complete Slang-to-LLVM example.
  """
  use Beaver.ComposerGenerator, prefix: "mlirCreateConversion"
  require Logger
  require MLIR.Conversion.Callbacks
  alias Beaver.MLIR

  defmodule Error do
    @moduledoc "An error returned by a failed dialect conversion."
    defexception [:mode, :result, diagnostics: [], callback_failure: nil, reason: :conversion_failed]

    @type t() :: %__MODULE__{
            mode: Beaver.MLIR.Conversion.mode(),
            result: Beaver.MLIR.Conversion.Result.t() | nil,
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

      if error.diagnostics == [], do: details, else: MLIR.Diagnostic.format(error.diagnostics, details)
    end
  end

  @type mode() :: :full | :partial | :analysis
  @type conversion_ir() :: MLIR.Module.t() | Operation.t()
  @type result() :: {:ok, conversion_ir(), [term()]} | {:ok, MLIR.Conversion.Result.t()} | {:error, Error.t()}
  @spec apply(
          mode(),
          conversion_ir(),
          ConversionTarget.t(),
          RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: result()
  def apply(mode, ir, target, patterns, opts \\ [])

  def apply(mode, ir, target, patterns, opts) do
    case opts[:profile] do
      nil ->
        run(mode, ir, target, patterns, opts, false)

      %Beaver.Execution.Profile{} = profile ->
        Beaver.Execution.Profile.measure(profile, :conversion, fn ->
          {result, _receipt} = run(mode, ir, target, patterns, opts, true)
          result
        end)
    end
  end

  @doc "Runs conversion with an explicit mutation and legalization result. Errors include the same report in `error.result`."
  def run(mode, ir, target, patterns, opts \\ []),
    do: apply(mode, ir, target, patterns, Keyword.put(opts, :report, true))

  @doc "Simulates legalization and rolls back every speculative IR change."
  def analyze(ir, target, patterns, opts \\ []), do: run(:analysis, ir, target, patterns, opts)

  @doc """
  Runs one conversion with bounded callback and native-worker profiling.

  Returns the ordinary conversion result together with a machine-readable `Beaver.MLIR.Conversion.Profile` receipt.
  Profiling is opt-in so normal conversion keeps its existing callback path and measurement overhead.
  """
  @spec profile(
          mode(),
          conversion_ir(),
          ConversionTarget.t(),
          RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: {result(), Beaver.MLIR.Conversion.Profile.receipt()}
  def profile(mode, ir, target, patterns, opts \\ []), do: run(mode, ir, target, patterns, opts, true)

  defp run(mode, ir, %ConversionTarget{} = target, %kind{} = patterns, opts, profile?)
       when mode in [:full, :partial, :analysis] and kind in [RewritePatternSet, MLIR.FrozenRewritePatternSet] do
    MLIR.AnalysisSession.track(opts[:analysis_session], fn ->
      options = conversion_options!(opts)
      owned = kind == RewritePatternSet
      patterns = if owned, do: RewritePatternSet.freeze(patterns), else: patterns
      do_apply(mode, ir, target, patterns, owned, options, profile?)
    end)
  end

  defp do_apply(
         mode,
         ir,
         %ConversionTarget{registration: registration},
         %MLIR.FrozenRewritePatternSet{ref: patterns_ref},
         owns_patterns,
         {folding_mode, materializations, timeout_ms, execution_profile, report?},
         profile?
       ) do
    operation = Operation.from_module(ir)
    context = MLIR.context(ir)
    profile = if profile?, do: Map.put(__MODULE__.Profile.start(mode, ir), :execution_profile, execution_profile)

    id =
      try do
        CAPI.beaver_raw_execution_worker(
          context.ref,
          :conversion,
          {%{full: 0, partial: 1, analysis: 2}[mode], registration, operation.ref, patterns_ref, owns_patterns,
           folding_mode, materializations, report?},
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
    unless Keyword.keyword?(opts), do: raise(ArgumentError, "conversion options must be a keyword list")

    unsupported =
      Keyword.keys(opts) -- [:timeout, :folding_mode, :build_materializations, :profile, :report, :analysis_session]

    if unsupported != [], do: raise(ArgumentError, "unsupported conversion options: #{inspect(unsupported)}")
    folding_mode = folding_mode!(Keyword.get(opts, :folding_mode))
    materializations = materializations!(Keyword.get(opts, :build_materializations))
    timeout_ms = Beaver.Execution.callback_timeout!(opts)
    report? = Keyword.get(opts, :report, false)
    unless is_boolean(report?), do: raise(ArgumentError, ":report must be boolean")
    {folding_mode, materializations, timeout_ms, opts[:profile], report?}
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

  defp await(id, mode, ir, timeout_ms, callback_failure, profile) do
    receive do
      {^id, {outcome, costs}} ->
        {result, native_profile} =
          case outcome do
            {:ok, value} -> value
            {:error, _reason} -> {nil, {0, 0}}
          end

        if profile && profile[:execution_profile] do
          Beaver.Execution.Profile.native(profile.execution_profile, :conversion, :worker, costs)
          Beaver.Execution.Profile.record(profile.execution_profile, :conversion_target_lock, elem(native_profile, 1))
        end

        conversion_result = finish(mode, ir, result, callback_failure)

        if profile,
          do: {conversion_result, __MODULE__.Profile.finish(profile, conversion_result, ir, native_profile)},
          else: conversion_result

      message when MLIR.Conversion.Callbacks.is_message(message) ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure, profile)
    after
      timeout_ms + 1_000 ->
        Logger.warning("still waiting for #{mode} dialect conversion to finish")
        await(id, mode, ir, timeout_ms, callback_failure, profile)
    end
  end

  defp handle_and_await(message, id, mode, ir, timeout_ms, callback_failure, profile) do
    started_at = if profile, do: CAPI.beaver_raw_execution_clock()
    {:handled, failure} = __MODULE__.Callbacks.handle(message)

    profile =
      if profile do
        stopped_at = CAPI.beaver_raw_execution_clock()
        service_ns = max(stopped_at - started_at, 0)
        native_wait_ns = max(CAPI.beaver_raw_execution_clock() - elem(message, 4), service_ns)

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

  defp finish(mode, ir, {json, diagnostics}, callback_failure) when is_binary(json) do
    {_, diagnostics} = Beaver.Native.check!({json, diagnostics})
    report = __MODULE__.Result.decode(json, mode, ir, MLIR.Diagnostic.process(diagnostics), callback_failure)

    if report.status == :succeeded,
      do: {:ok, report},
      else:
        {:error,
         %Error{mode: mode, result: report, diagnostics: report.diagnostics, callback_failure: callback_failure}}
  end

  defp finish(mode, ir, result, callback_failure) do
    {logical_result, diagnostics} = Beaver.Native.check!(result)

    if MLIR.LogicalResult.success?(logical_result) and is_nil(callback_failure),
      do: {:ok, ir, diagnostics},
      else: {:error, %Error{mode: mode, diagnostics: diagnostics, callback_failure: callback_failure}}
  end
end
