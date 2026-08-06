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

      if error.diagnostics == [] do
        details
      else
        MLIR.Diagnostic.format(error.diagnostics, details)
      end
    end
  end

  @type mode() :: :full | :partial
  @type conversion_ir() :: MLIR.Module.t() | MLIR.Operation.t()
  @type result() :: {:ok, conversion_ir(), [term()]} | {:error, Error.t()}

  @spec apply(
          mode(),
          conversion_ir(),
          MLIR.ConversionTarget.t(),
          MLIR.RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: result()
  def apply(mode, ir, target, patterns, opts \\ [])

  def apply(mode, ir, %MLIR.ConversionTarget{} = target, %MLIR.RewritePatternSet{} = set, opts)
      when mode in [:full, :partial] do
    {folding_mode, materializations, timeout_ms} = conversion_options!(opts)
    frozen = MLIR.RewritePatternSet.freeze(set)

    do_apply(
      mode,
      ir,
      target,
      frozen,
      true,
      folding_mode,
      materializations,
      timeout_ms
    )
  end

  def apply(
        mode,
        ir,
        %MLIR.ConversionTarget{registration: registration},
        %MLIR.FrozenRewritePatternSet{ref: patterns_ref},
        opts
      )
      when mode in [:full, :partial] do
    {folding_mode, materializations, timeout_ms} = conversion_options!(opts)

    do_apply(
      mode,
      ir,
      %MLIR.ConversionTarget{registration: registration},
      %MLIR.FrozenRewritePatternSet{ref: patterns_ref},
      false,
      folding_mode,
      materializations,
      timeout_ms
    )
  end

  defp do_apply(
         mode,
         ir,
         %MLIR.ConversionTarget{registration: registration},
         %MLIR.FrozenRewritePatternSet{ref: patterns_ref},
         owns_patterns,
         folding_mode,
         materializations,
         timeout_ms
       ) do
    operation = operation(ir)
    context = MLIR.context(ir)

    id =
      try do
        MLIR.CAPI.beaver_raw_apply_conversion_async(
          mode == :full,
          registration,
          operation.ref,
          patterns_ref,
          owns_patterns,
          folding_mode,
          materializations
        )
      rescue
        exception ->
          if owns_patterns do
            MLIR.FrozenRewritePatternSet.threaded_destroy(
              context,
              %MLIR.FrozenRewritePatternSet{ref: patterns_ref}
            )
          end

          reraise exception, __STACKTRACE__
      end

    await(id, mode, ir, timeout_ms, nil)
  end

  defp conversion_options!(opts) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "conversion options must be a keyword list"
    end

    unsupported = Keyword.keys(opts) -- [:timeout, :folding_mode, :build_materializations]

    if unsupported != [] do
      raise ArgumentError, "unsupported conversion options: #{inspect(unsupported)}"
    end

    folding_mode = folding_mode!(Keyword.get(opts, :folding_mode))
    materializations = materializations!(Keyword.get(opts, :build_materializations))

    timeout_ms = Keyword.get(opts, :timeout, 30_000)

    unless is_integer(timeout_ms) and timeout_ms >= 0 do
      raise ArgumentError, ":timeout must be a non-negative integer"
    end

    {folding_mode, materializations, timeout_ms}
  end

  defp folding_mode!(nil), do: -1
  defp folding_mode!(:never), do: 0
  defp folding_mode!(:before_patterns), do: 1
  defp folding_mode!(:after_patterns), do: 2

  defp folding_mode!(mode) do
    raise ArgumentError, "unsupported conversion folding mode: #{inspect(mode)}"
  end

  defp materializations!(nil), do: -1
  defp materializations!(false), do: 0
  defp materializations!(true), do: 1

  defp materializations!(value) do
    raise ArgumentError, "build_materializations must be boolean, got: #{inspect(value)}"
  end

  @spec full(
          conversion_ir(),
          MLIR.ConversionTarget.t(),
          MLIR.RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: result()
  def full(ir, target, patterns, opts \\ []), do: apply(:full, ir, target, patterns, opts)

  @spec partial(
          conversion_ir(),
          MLIR.ConversionTarget.t(),
          MLIR.RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: result()
  def partial(ir, target, patterns, opts \\ []), do: apply(:partial, ir, target, patterns, opts)

  @spec apply!(
          mode(),
          conversion_ir(),
          MLIR.ConversionTarget.t(),
          MLIR.RewritePatternSet.t() | MLIR.FrozenRewritePatternSet.t(),
          keyword()
        ) :: conversion_ir()
  def apply!(mode, ir, target, patterns, opts \\ []) do
    case apply(mode, ir, target, patterns, opts) do
      {:ok, converted, _diagnostics} -> converted
      {:error, error} -> raise error
    end
  end

  defp operation(%MLIR.Operation{} = operation), do: operation
  defp operation(%MLIR.Module{} = module), do: MLIR.Operation.from_module(module)

  defp await(id, mode, ir, timeout_ms, callback_failure) do
    receive do
      {:conversion_done, ^id, result} ->
        finish(mode, ir, result, callback_failure)

      {name, _token, _callback, _callback_id, _arg} = message
      when name in [:conversion_legality, :convert_type, :convert_types] ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure)

      {name, _token, _callback, _callback_id, _arg1, _arg2, _arg3} = message
      when name in [:conversion_pattern, :conversion_pattern_1_to_n] ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure)

      {:source_materialization, _token, _callback, _callback_id, _rewriter, _type, _inputs, _loc} =
          message ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure)

      {:target_materialization, _token, _callback, _callback_id, _rewriter, _type, _inputs, _loc,
       _original_type} = message ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure)

      {:target_materialization_1_to_n, _token, _callback, _callback_id, _rewriter, _output_types,
       _inputs, _loc, _original_type} = message ->
        handle_and_await(message, id, mode, ir, timeout_ms, callback_failure)
    after
      timeout_ms + 1_000 ->
        Logger.warning("still waiting for #{mode} dialect conversion to finish")
        await(id, mode, ir, timeout_ms, callback_failure)
    end
  end

  defp handle_and_await(message, id, mode, ir, timeout_ms, callback_failure) do
    {:handled, failure} = __MODULE__.Callbacks.handle(message)
    await(id, mode, ir, timeout_ms, callback_failure || failure)
  end

  defp finish(mode, _ir, nil, callback_failure) do
    {:error,
     %Error{mode: mode, callback_failure: callback_failure, reason: :native_worker_failed}}
  end

  defp finish(mode, ir, result, callback_failure) do
    {logical_result, diagnostics} = Beaver.Native.check!(result)

    if MLIR.LogicalResult.success?(logical_result) and is_nil(callback_failure) do
      {:ok, ir, diagnostics}
    else
      {:error,
       %Error{
         mode: mode,
         diagnostics: diagnostics,
         callback_failure: callback_failure
       }}
    end
  end
end

defmodule Beaver.MLIR.Conversion.Callbacks do
  @moduledoc false

  alias Beaver.MLIR
  alias Kinda.CallbackRuntime

  def handle({:conversion_legality, token, callback, _id, operation}) do
    invoke(token, fn -> legality(callback.(native(operation))) end, &reply_legality/2)
  end

  def handle({:convert_type, token, callback, _id, type}) do
    invoke(token, fn -> converted_type(callback.(native(type))) end, &reply_type/2)
  end

  def handle({:convert_types, token, callback, _id, type}) do
    invoke(token, fn -> converted_types(callback.(native(type))) end, &reply_types/2)
  end

  def handle({:source_materialization, token, callback, _id, rewriter, type, inputs, loc}) do
    invoke(
      token,
      fn ->
        materialized(callback.(native(rewriter), native(type), native(inputs), native(loc)))
      end,
      &reply_value/2
    )
  end

  def handle(
        {:target_materialization, token, callback, _id, rewriter, type, inputs, loc,
         original_type}
      ) do
    invoke(
      token,
      fn ->
        materialized(
          callback.(
            native(rewriter),
            native(type),
            native(inputs),
            native(loc),
            native(original_type)
          )
        )
      end,
      &reply_value/2
    )
  end

  def handle(
        {:target_materialization_1_to_n, token, callback, _id, rewriter, output_types, inputs,
         loc, original_type}
      ) do
    invoke(
      token,
      fn ->
        materialized_values(
          callback.(
            native(rewriter),
            native(output_types),
            native(inputs),
            native(loc),
            native(original_type)
          )
        )
      end,
      &reply_values/2
    )
  end

  def handle({:conversion_pattern, token, callback, _id, operation, operands, rewriter}) do
    invoke(
      token,
      fn -> pattern_result(callback.(native(operation), native(operands), native(rewriter))) end,
      &reply_pattern/2
    )
  end

  def handle({:conversion_pattern_1_to_n, token, callback, _id, operation, ranges, rewriter}) do
    invoke(
      token,
      fn -> pattern_result(callback.(native(operation), native(ranges), native(rewriter))) end,
      &reply_pattern/2
    )
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

  defp legality(other),
    do: raise(ArgumentError, "invalid dynamic legality result: #{inspect(other)}")

  defp converted_type(%MLIR.Type{} = type), do: {:ok, {:success, type}}
  defp converted_type({:ok, %MLIR.Type{} = type}), do: {:ok, {:success, type}}
  defp converted_type(:declined), do: {:ok, :declined}
  defp converted_type({:error, _reason} = error), do: error

  defp converted_type(other),
    do: raise(ArgumentError, "invalid type conversion result: #{inspect(other)}")

  defp converted_types(types) when is_list(types) do
    validate_list!(types, MLIR.Type, "1:N type conversion")
    {:ok, {:success, types}}
  end

  defp converted_types({:ok, types}) when is_list(types), do: converted_types(types)
  defp converted_types(:declined), do: {:ok, :declined}
  defp converted_types({:error, _reason} = error), do: error

  defp converted_types(other),
    do: raise(ArgumentError, "invalid 1:N conversion result: #{inspect(other)}")

  defp materialized(%MLIR.Value{} = value), do: {:ok, {:success, value}}
  defp materialized({:ok, %MLIR.Value{} = value}), do: {:ok, {:success, value}}
  defp materialized(nil), do: {:ok, :declined}
  defp materialized(:declined), do: {:ok, :declined}
  defp materialized({:error, _reason} = error), do: error

  defp materialized(other),
    do: raise(ArgumentError, "invalid materialization result: #{inspect(other)}")

  defp materialized_values(values) when is_list(values) do
    validate_list!(values, MLIR.Value, "1:N target materialization")
    {:ok, {:success, values}}
  end

  defp materialized_values({:ok, values}) when is_list(values), do: materialized_values(values)
  defp materialized_values(:declined), do: {:ok, :declined}
  defp materialized_values({:error, _reason} = error), do: error

  defp materialized_values(other),
    do: raise(ArgumentError, "invalid 1:N materialization result: #{inspect(other)}")

  defp pattern_result(:ok), do: {:ok, :match}
  defp pattern_result(true), do: {:ok, :match}
  defp pattern_result({:ok, _state}), do: {:ok, :match}
  defp pattern_result(:no_match), do: {:ok, :no_match}
  defp pattern_result(false), do: {:ok, :no_match}
  defp pattern_result({:error, _reason} = error), do: error

  defp pattern_result(other),
    do: raise(ArgumentError, "invalid conversion pattern result: #{inspect(other)}")

  defp validate_list!(values, module, label) do
    unless Enum.all?(values, &is_struct(&1, module)) do
      raise ArgumentError, "#{label} must return only #{inspect(module)} values"
    end
  end

  defp reply_legality(token, {:ok, result}) do
    code = %{legal: 0, illegal: 1, no_opinion: 2} |> Map.fetch!(result)
    MLIR.CAPI.beaver_raw_callback_reply_code(token, true, code)
  end

  defp reply_legality(token, _failure),
    do: MLIR.CAPI.beaver_raw_callback_reply_code(token, false, 1)

  defp reply_type(token, {:ok, {:success, %MLIR.Type{ref: ref}}}),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_callback(token, true, 0, ref)

  defp reply_type(token, {:ok, :declined}),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_callback(token, true, 2, nil)

  defp reply_type(token, _failure),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_callback(token, false, 1, nil)

  defp reply_types(token, {:ok, {:success, types}}),
    do:
      MLIR.CAPI.beaver_raw_type_converter_reply_types(
        token,
        true,
        0,
        Enum.map(types, & &1.ref)
      )

  defp reply_types(token, {:ok, :declined}),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_types(token, true, 2, [])

  defp reply_types(token, _failure),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_types(token, false, 1, [])

  defp reply_value(token, {:ok, {:success, %MLIR.Value{ref: ref}}}),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_value(token, true, ref)

  defp reply_value(token, {:ok, :declined}),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_value(token, true, nil)

  defp reply_value(token, _failure),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_value(token, false, nil)

  defp reply_values(token, {:ok, {:success, values}}),
    do:
      MLIR.CAPI.beaver_raw_type_converter_reply_values(
        token,
        true,
        0,
        Enum.map(values, & &1.ref)
      )

  defp reply_values(token, _failure),
    do: MLIR.CAPI.beaver_raw_type_converter_reply_values(token, false, 1, [])

  defp reply_pattern(token, {:ok, :match}),
    do: MLIR.CAPI.beaver_raw_callback_reply(token, true)

  defp reply_pattern(token, _no_match_or_failure),
    do: MLIR.CAPI.beaver_raw_callback_reply(token, false)

  defp callback_failure({:error, reason}), do: {:error, reason}

  defp callback_failure({:exception, _kind, _reason, _stacktrace} = exception),
    do: exception

  defp callback_failure(_outcome), do: nil
end
