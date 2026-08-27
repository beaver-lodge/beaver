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

  defmodule FixedPoint do
    @moduledoc """
    A structured pass pipeline that repeats until its IR fingerprint converges.

    The declaration is inert and owns no native resources. `Beaver.Composer`
    materializes it inside the target context.
    """

    @enforce_keys [:name, :pipeline, :max_iterations, :on_convergence_failure]
    defstruct [:name, :pipeline, :max_iterations, :on_convergence_failure]

    @type failure_action :: :warn | :error | :silent
    @type t :: %__MODULE__{
            name: String.t(),
            pipeline: list(),
            max_iterations: pos_integer(),
            on_convergence_failure: failure_action()
          }
  end

  @doc """
  Declares a structured fixed-point pipeline.

  Failure to converge is an error by default. Choose `:warn` or `:silent`
  explicitly only when continuing with a partially converged IR is intended.
  """
  @spec composite_fixed_point(keyword()) :: FixedPoint.t()
  def composite_fixed_point(opts) when is_list(opts) do
    opts = validate_fixed_point_options!(opts)

    %FixedPoint{
      name: opts |> Keyword.get(:name, "CompositeFixedPointPass") |> validate_fixed_point_name!(),
      pipeline: opts |> Keyword.fetch!(:pipeline) |> validate_fixed_point_pipeline!(),
      max_iterations:
        opts |> Keyword.get(:max_iterations, 10) |> validate_fixed_point_max_iterations!(),
      on_convergence_failure:
        opts
        |> Keyword.get(:on_convergence_failure, :error)
        |> validate_convergence_failure_action!()
    }
  end

  def composite_fixed_point(other) do
    raise ArgumentError, "fixed-point options must be a keyword list, got: #{inspect(other)}"
  end

  defp validate_fixed_point_options!(opts) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "fixed-point options must be a keyword list"
    end

    supported = [:name, :pipeline, :max_iterations, :on_convergence_failure]

    case Keyword.keys(opts) -- supported do
      [] -> :ok
      keys -> raise ArgumentError, "unsupported fixed-point options: #{inspect(keys)}"
    end

    opts
  end

  defp validate_fixed_point_name!(name) do
    unless is_binary(name) and name != "" do
      raise ArgumentError, ":name must be a non-empty string"
    end

    name
  end

  defp validate_fixed_point_pipeline!(pipeline) do
    unless is_list(pipeline) and pipeline != [] do
      raise ArgumentError, ":pipeline must be a non-empty Composer pass list"
    end

    pipeline
  end

  defp validate_fixed_point_max_iterations!(max_iterations) do
    unless is_integer(max_iterations) and max_iterations > 0 do
      raise ArgumentError, ":max_iterations must be a positive integer"
    end

    max_iterations
  end

  defp validate_convergence_failure_action!(failure_action) do
    unless failure_action in [:warn, :error, :silent] do
      raise ArgumentError,
            ":on_convergence_failure must be :warn, :error, or :silent"
    end

    failure_action
  end

  @doc "Returns whether the linked LLVM can configure convergence failure behavior."
  @spec composite_fixed_point_failure_action_supported?() :: boolean()
  def composite_fixed_point_failure_action_supported? do
    MLIR.CAPI.beaverCompositeFixedPointFailureActionSupported()
    |> Beaver.Native.to_term()
  end

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

  @doc "Whether the linked LLVM supports packed tile-size and interchange parameters."
  @spec packed_params_supported?() :: boolean()
  def packed_params_supported? do
    MLIR.CAPI.beaverTransformPackedParamsSupported()
    |> Beaver.Native.to_term()
  end

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
