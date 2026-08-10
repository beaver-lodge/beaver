defmodule Beaver.MLIR.Transform do
  @moduledoc """
  Transform dialect execution and transformations MLIR provides by default.

  `apply_named_sequence/3` executes a named Transform dialect sequence without
  routing it through a pass manager. Transform IR may be supplied as a module,
  an operation nested in a module, textual MLIR, bytecode, or a resolved tuning
  schedule produced by `Beaver.MLIR.Transform.Schedule`.

  Handles remain owned by MLIR's transform state and never escape this call.
  Keeping expensive checks enabled (the default) makes use-after-consume handle
  errors visible as processed diagnostics.
  """

  alias __MODULE__.Schedule
  alias Beaver.MLIR

  use Beaver.ComposerGenerator, prefix: "mlirCreateTransforms"
  use Beaver.ComposerGenerator, prefix: "mlirCreateLinalg"
  use Beaver.ComposerGenerator, prefix: "mlirCreateGPU"

  defmodule Result do
    @moduledoc "The result of successfully applying a named transform sequence."

    @enforce_keys [:payload, :sequence, :diagnostics]
    defstruct [:payload, :sequence, :diagnostics]

    @type t() :: %__MODULE__{
            payload: MLIR.Module.t() | MLIR.Operation.t(),
            sequence: String.t(),
            diagnostics: [term()]
          }
  end

  defmodule Error do
    @moduledoc "A structured Transform dialect parsing, resolution, or execution failure."

    defexception [:kind, :reason, diagnostics: [], sequence: nil]

    @type kind() ::
            :invalid_schedule
            | :silenceable_failure
            | :definite_failure
            | :evaluation_failure
            | :constraint_failure

    @type t() :: %__MODULE__{
            kind: kind(),
            reason: term(),
            diagnostics: [term()],
            sequence: String.t() | nil
          }

    @impl true
    def message(%__MODULE__{} = error) do
      prefix =
        case error.kind do
          :invalid_schedule -> "invalid transform schedule"
          :silenceable_failure -> "transform sequence failed silenceably"
          :definite_failure -> "transform sequence failed definitively"
          :evaluation_failure -> "transform schedule evaluation failed"
          :constraint_failure -> "transform schedule constraints failed"
        end

      prefix =
        if error.sequence do
          prefix <> " (#{error.sequence})"
        else
          prefix
        end

      details = if is_nil(error.reason), do: prefix, else: prefix <> ": " <> inspect(error.reason)

      if error.diagnostics == [] do
        details
      else
        MLIR.Diagnostic.format(error.diagnostics, details)
      end
    end
  end

  @type schedule_input() ::
          MLIR.Module.t()
          | MLIR.Operation.t()
          | binary()
          | Schedule.Resolved.t()

  @type execution_option() ::
          {:sequence, String.t()}
          | {:expensive_checks, boolean()}
          | {:enforce_single_top_level_transform_op, boolean()}

  @doc """
  Applies a named Transform dialect sequence to a payload operation or module.

  The default sequence is `__transform_main`. LLVM's expensive handle checks
  and single-top-level-transform enforcement both default to `true` and can be
  configured independently.
  """
  @spec apply_named_sequence(
          MLIR.Module.t() | MLIR.Operation.t(),
          schedule_input(),
          [execution_option()]
        ) :: {:ok, Result.t()} | {:error, Error.t()}
  def apply_named_sequence(payload, schedule, opts \\ []) do
    with :ok <- validate_execution_options(opts),
         payload_operation <- MLIR.Operation.from_module(payload),
         context <- MLIR.context(payload_operation) do
      Schedule.with_module(schedule, context, fn schedule_module ->
        execute_loaded_schedule(payload, payload_operation, schedule, schedule_module, opts)
      end)
    end
  rescue
    exception in [ArgumentError] ->
      {:error,
       %Error{
         kind: :invalid_schedule,
         reason: Exception.message(exception),
         sequence: Keyword.get(opts, :sequence)
       }}
  end

  defp execute_loaded_schedule(payload, payload_operation, schedule, schedule_module, opts) do
    sequence = Keyword.get(opts, :sequence, Schedule.sequence(schedule))

    case Schedule.find_sequence(schedule_module, sequence) do
      {:ok, transform_root} ->
        apply_with_options(
          payload,
          payload_operation,
          schedule_module,
          transform_root,
          sequence,
          opts
        )

      {:error, error} ->
        {:error, error}
    end
  end

  @doc "Bang variant of `apply_named_sequence/3`."
  @spec apply_named_sequence!(
          MLIR.Module.t() | MLIR.Operation.t(),
          schedule_input(),
          [execution_option()]
        ) :: MLIR.Module.t() | MLIR.Operation.t()
  def apply_named_sequence!(payload, schedule, opts \\ []) do
    case apply_named_sequence(payload, schedule, opts) do
      {:ok, %Result{payload: transformed}} -> transformed
      {:error, error} -> raise error
    end
  end

  @doc "Alias for `apply_named_sequence/3`."
  def execute(payload, schedule, opts \\ []), do: apply_named_sequence(payload, schedule, opts)

  @doc "Alias for `apply_named_sequence!/3`."
  def execute!(payload, schedule, opts \\ []), do: apply_named_sequence!(payload, schedule, opts)

  @doc "Creates Transform dialect's `!transform.any_op` type."
  def any_op_type(opts \\ []) do
    Beaver.Deferred.from_opts(opts, &MLIR.CAPI.mlirTransformAnyOpTypeGet/1)
  end

  @doc "Creates Transform dialect's `!transform.any_value` type."
  def any_value_type(opts \\ []) do
    Beaver.Deferred.from_opts(opts, &MLIR.CAPI.mlirTransformAnyValueTypeGet/1)
  end

  @doc "Creates Transform dialect's `!transform.any_param` type."
  def any_param_type(opts \\ []) do
    Beaver.Deferred.from_opts(opts, &MLIR.CAPI.mlirTransformAnyParamTypeGet/1)
  end

  @doc "Creates an operation-specific Transform handle type."
  def operation_type(operation_name, opts \\ []) when is_binary(operation_name) do
    Beaver.Deferred.from_opts(opts, fn context ->
      MLIR.CAPI.mlirTransformOperationTypeGet(
        context,
        MLIR.StringRef.create(operation_name)
      )
    end)
  end

  @doc "Creates a Transform parameter type wrapping an MLIR type."
  def param_type(type) do
    case type do
      %MLIR.Type{} ->
        MLIR.CAPI.mlirTransformParamTypeGet(MLIR.context(type), type)

      %Beaver.Deferred{} = deferred ->
        Beaver.Deferred.defer(&param_type(Beaver.Deferred.resolve(deferred, &1)))
    end
  end

  defp apply_with_options(
         payload,
         payload_operation,
         schedule_module,
         transform_root,
         sequence,
         opts
       ) do
    context = MLIR.context(payload_operation)
    transform_options = MLIR.CAPI.mlirTransformOptionsCreate()

    try do
      MLIR.CAPI.mlirTransformOptionsEnableExpensiveChecks(
        transform_options,
        Keyword.get(opts, :expensive_checks, true)
      )

      MLIR.CAPI.mlirTransformOptionsEnforceSingleTopLevelTransformOp(
        transform_options,
        Keyword.get(opts, :enforce_single_top_level_transform_op, true)
      )

      {logical_result, diagnostics} =
        MLIR.CAPI.mlirTransformApplyNamedSequenceWithDiagnostics(
          context,
          payload_operation,
          transform_root,
          MLIR.Operation.from_module(schedule_module),
          transform_options
        )

      diagnostics = MLIR.Diagnostic.process(diagnostics)

      if MLIR.LogicalResult.success?(logical_result) do
        {:ok, %Result{payload: payload, sequence: sequence, diagnostics: diagnostics}}
      else
        {:error,
         %Error{
           kind: classify_failure(diagnostics),
           reason: :transform_application_failed,
           diagnostics: diagnostics,
           sequence: sequence
         }}
      end
    after
      MLIR.CAPI.mlirTransformOptionsDestroy(transform_options)
    end
  end

  defp classify_failure(diagnostics) do
    if Enum.any?(diagnostics, &definite_diagnostic?/1) do
      :definite_failure
    else
      :silenceable_failure
    end
  end

  # The upstream C entry point intentionally flattens DiagnosedSilenceableFailure
  # to LogicalResult. Definite failures retain stable diagnostic wording; all
  # other execution failures are the propagated silenceable class.
  defp definite_diagnostic?({_severity, _location, message, nested}) do
    message = String.downcase(message)

    String.contains?(message, "definite failure") or
      String.contains?(message, "non-deterministic choice") or
      String.contains?(message, "callback raised") or
      String.contains?(message, "callback failed") or
      String.contains?(message, "callback returned an invalid") or
      Enum.any?(nested, &definite_diagnostic?/1)
  end

  defp validate_execution_options(opts) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "transform execution options must be a keyword list"
    end

    supported = [
      :sequence,
      :expensive_checks,
      :enforce_single_top_level_transform_op
    ]

    case Keyword.keys(opts) -- supported do
      [] ->
        for key <- [:expensive_checks, :enforce_single_top_level_transform_op],
            value = Keyword.get(opts, key, true),
            not is_boolean(value) do
          raise ArgumentError, "#{inspect(key)} must be boolean, got: #{inspect(value)}"
        end

        case Keyword.get(opts, :sequence, "__transform_main") do
          sequence when is_binary(sequence) and sequence != "" ->
            :ok

          sequence ->
            raise ArgumentError, ":sequence must be a non-empty string, got: #{inspect(sequence)}"
        end

      unsupported ->
        raise ArgumentError, "unsupported transform execution options: #{inspect(unsupported)}"
    end
  end
end
