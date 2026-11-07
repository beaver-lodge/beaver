alias Beaver.MLIR.{CAPI, Operation}
alias Beaver.Deferred

defmodule Beaver.MLIR.Transform do
  @moduledoc """
  Executes named Transform dialect sequences directly and exposes MLIR's
  default transformations. Schedules accept modules, nested operations, text
  or bytecode. Transform handles never escape execution; expensive checks
  (enabled by default) diagnose use-after-consume errors.
  """

  alias __MODULE__.{Schedule, Prepared}
  alias Beaver.MLIR

  use Beaver.ComposerGenerator, prefix: "mlirCreateTransforms"
  use Beaver.ComposerGenerator, prefix: "mlirCreateLinalg"
  use Beaver.ComposerGenerator, prefix: "mlirCreateGPU"

  defmodule FixedPoint do
    @moduledoc """
    A pipeline repeated until its IR fingerprint converges. This declaration
    owns no native resources; `Beaver.Composer` materializes it in the target context.
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
  Declares a fixed-point pipeline. Non-convergence fails by default;
  `:warn` or `:silent` explicitly allow partially converged IR.
  """
  @spec composite_fixed_point(keyword()) :: FixedPoint.t()
  def composite_fixed_point(opts) when is_list(opts) do
    opts = validate_fixed_point_options!(opts)

    %FixedPoint{
      name: fixed_point_value!(:name, Keyword.get(opts, :name, "CompositeFixedPointPass")),
      pipeline: fixed_point_value!(:pipeline, Keyword.fetch!(opts, :pipeline)),
      max_iterations: fixed_point_value!(:max_iterations, Keyword.get(opts, :max_iterations, 10)),
      on_convergence_failure:
        fixed_point_value!(:on_convergence_failure, Keyword.get(opts, :on_convergence_failure, :error))
    }
  end

  def composite_fixed_point(other),
    do: raise(ArgumentError, "fixed-point options must be a keyword list, got: #{inspect(other)}")

  defp validate_fixed_point_options!(opts) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "fixed-point options must be a keyword list")

    supported = [:name, :pipeline, :max_iterations, :on_convergence_failure]

    case Keyword.keys(opts) -- supported do
      [] -> :ok
      keys -> raise ArgumentError, "unsupported fixed-point options: #{inspect(keys)}"
    end

    opts
  end

  defp fixed_point_value!(:name, value) when is_binary(value) and value != "", do: value
  defp fixed_point_value!(:pipeline, value) when is_list(value) and value != [], do: value
  defp fixed_point_value!(:max_iterations, value) when is_integer(value) and value > 0, do: value
  defp fixed_point_value!(:on_convergence_failure, value) when value in [:warn, :error, :silent], do: value

  defp fixed_point_value!(key, _value) do
    expected = %{
      name: "a non-empty string",
      pipeline: "a non-empty Composer pass list",
      max_iterations: "a positive integer",
      on_convergence_failure: ":warn, :error, or :silent"
    }

    raise ArgumentError, "#{inspect(key)} must be #{Map.fetch!(expected, key)}"
  end

  @doc "Returns whether the linked LLVM can configure convergence failure behavior."
  @spec composite_fixed_point_failure_action_supported?() :: boolean()
  def composite_fixed_point_failure_action_supported?,
    do: true

  defmodule Result do
    @moduledoc "The result of successfully applying a named transform sequence."

    @enforce_keys [:payload, :sequence, :diagnostics]
    defstruct [:payload, :sequence, :diagnostics]

    @type t() :: %__MODULE__{payload: MLIR.Module.t() | Operation.t(), sequence: String.t(), diagnostics: [term()]}
  end

  defmodule Error do
    @moduledoc "A structured Transform dialect parsing, resolution, or execution failure."

    defexception [:kind, :reason, diagnostics: [], sequence: nil]

    @type kind() ::
            :invalid_schedule | :silenceable_failure | :definite_failure | :evaluation_failure | :constraint_failure

    @type t() :: %__MODULE__{kind: kind(), reason: term(), diagnostics: [term()], sequence: String.t() | nil}

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

      prefix = if error.sequence, do: prefix <> " (#{error.sequence})", else: prefix

      details = if is_nil(error.reason), do: prefix, else: prefix <> ": " <> inspect(error.reason)

      if error.diagnostics == [],
        do: details,
        else: MLIR.Diagnostic.format(error.diagnostics, details)
    end
  end

  @type schedule_input() ::
          Prepared.t() | MLIR.Module.t() | Operation.t() | binary() | {:text, binary()} | {:bytecode, binary()}

  @type execution_option() ::
          {:sequence, String.t()}
          | {:expensive_checks, boolean()}
          | {:enforce_single_top_level_transform_op, boolean()}
          | {:analysis_session, MLIR.AnalysisSession.t()}

  @doc "Whether the linked LLVM supports packed tile-size and interchange parameters."
  @spec packed_params_supported?() :: boolean()
  def packed_params_supported?,
    do: true

  @doc """
  Applies a named sequence (default `__transform_main`) to an operation or module.
  Expensive handle checks and single-top-level-transform enforcement both
  default to `true` and can be configured independently.
  """
  @spec apply_named_sequence(MLIR.Module.t() | Operation.t(), schedule_input(), [execution_option()]) ::
          {:ok, Result.t()} | {:error, Error.t()}
  def apply_named_sequence(payload, schedule, opts \\ [])

  def apply_named_sequence(payload, %Prepared{} = prepared, opts) do
    validate_execution_options(opts)
    if prepared.owner != self(), do: raise(ArgumentError, "prepared transform belongs to another process")

    if Keyword.get(opts, :sequence, prepared.sequence) != prepared.sequence,
      do: raise(ArgumentError, "prepared transform entry sequence cannot be changed")

    operation = Operation.from_module(payload)

    Beaver.Execution.run(
      MLIR.context(operation),
      :transform,
      {prepared.ref, operation.ref, Keyword.get(opts, :expensive_checks, true),
       Keyword.get(opts, :enforce_single_top_level_transform_op, true)},
      analysis_session: opts[:analysis_session]
    )
    |> transform_result(payload, prepared.sequence)
  rescue
    exception in [ArgumentError] -> invalid_schedule(exception, prepared.sequence)
  end

  def apply_named_sequence(payload, schedule, opts) do
    with {:ok, prepared} <- prepare(schedule, MLIR.context(payload), opts) do
      Beaver.Native.with_resource(prepared, &close/1, &apply_named_sequence(payload, &1, opts))
    end
  rescue
    exception in [ArgumentError] -> invalid_schedule(exception, Keyword.get(opts, :sequence))
  end

  @doc "Parses and verifies a schedule once and resolves its entry sequence. The context must outlive the result."
  def prepare(schedule, %MLIR.Context{} = context, opts \\ []) do
    validate_execution_options(opts)
    sequence = Keyword.get(opts, :sequence, "__transform_main")

    Schedule.load(schedule, context, fn module ->
      with {:ok, root} <- Schedule.find_sequence(module, sequence) do
        ref = CAPI.beaver_raw_transform_prepare(module.ref, root.ref)
        {:ok, %Prepared{ref: ref, context: context, sequence: sequence, owner: self()}}
      end
    end)
  rescue
    exception in [ArgumentError] -> invalid_schedule(exception, Keyword.get(opts, :sequence))
  end

  @doc "Closes a prepared schedule. Repeated close is harmless; further execution returns an error."
  def close(%Prepared{ref: ref, owner: owner}) when owner == self(), do: CAPI.beaver_raw_transform_close(ref)
  def close(%Prepared{}), do: raise(ArgumentError, "prepared transform belongs to another process")

  defp invalid_schedule(exception, sequence),
    do: {:error, %Error{kind: :invalid_schedule, reason: Exception.message(exception), sequence: sequence}}

  @doc "Bang variant of `apply_named_sequence/3`."
  @spec apply_named_sequence!(MLIR.Module.t() | Operation.t(), schedule_input(), [execution_option()]) ::
          MLIR.Module.t() | Operation.t()
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

  for kind <- [:op, :value, :param] do
    @doc "Creates Transform dialect's `!transform.any_#{kind}` type."
    def unquote(:"any_#{kind}_type")(opts \\ []),
      do: Deferred.from_opts(opts, &(CAPI.unquote(:"mlirTransformAny#{Macro.camelize(to_string(kind))}TypeGet") / 1))
  end

  @doc "Creates an operation-specific Transform handle type."
  def operation_type(op_name, opts \\ []) when is_binary(op_name) do
    Deferred.from_opts(opts, fn context ->
      CAPI.mlirTransformOperationTypeGet(context, MLIR.StringRef.create(op_name))
    end)
  end

  @doc "Creates a Transform parameter type wrapping an MLIR type."
  def param_type(type) do
    case type do
      %MLIR.Type{} -> CAPI.mlirTransformParamTypeGet(MLIR.context(type), type)
      %Deferred{} = deferred -> Deferred.defer(&param_type(Deferred.resolve(deferred, &1)))
    end
  end

  defp transform_result({logical_result, diagnostics}, payload, sequence) do
    diagnostics = MLIR.Diagnostic.process(diagnostics)

    if MLIR.LogicalResult.success?(logical_result),
      do: {:ok, %Result{payload: payload, sequence: sequence, diagnostics: diagnostics}},
      else:
        {:error,
         %Error{
           kind: classify_failure(diagnostics),
           reason: :transform_application_failed,
           diagnostics: diagnostics,
           sequence: sequence
         }}
  end

  defp classify_failure(diagnostics),
    do: if(Enum.any?(diagnostics, &definite_diagnostic?/1), do: :definite_failure, else: :silenceable_failure)

  # The upstream C entry point intentionally flattens DiagnosedSilenceableFailure
  # to LogicalResult. Definite failures retain stable diagnostic wording; all
  # other execution failures are the propagated silenceable class.
  defp definite_diagnostic?({_severity, _location, message, nested}) do
    message = String.downcase(message)

    String.contains?(message, [
      "definite failure",
      "non-deterministic choice",
      "callback raised",
      "callback failed",
      "callback returned an invalid"
    ]) or Enum.any?(nested, &definite_diagnostic?/1)
  end

  defp validate_execution_options(opts) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "transform execution options must be a keyword list")

    supported = [:sequence, :expensive_checks, :enforce_single_top_level_transform_op, :analysis_session]

    unsupported = Keyword.keys(opts) -- supported
    if unsupported != [], do: raise(ArgumentError, "unsupported transform execution options: #{inspect(unsupported)}")

    for key <- [:expensive_checks, :enforce_single_top_level_transform_op],
        value = Keyword.get(opts, key, true),
        not is_boolean(value),
        do: raise(ArgumentError, "#{inspect(key)} must be boolean, got: #{inspect(value)}")

    case Keyword.get(opts, :sequence, "__transform_main") do
      sequence when is_binary(sequence) and sequence != "" -> :ok
      sequence -> raise ArgumentError, ":sequence must be a non-empty string, got: #{inspect(sequence)}"
    end
  end
end
