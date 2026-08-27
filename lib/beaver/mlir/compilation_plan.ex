defmodule Beaver.MLIR.CompilationPlan do
  @moduledoc """
  A reusable, inspectable, cache-stable MLIR compiler declaration.

  A plan closes Composer-compatible pass data, a Transform schedule, target and
  schema configuration, bytecode version, context options, and telemetry metadata
  into one value. It owns no MLIR context or native resource.

  Pipeline strings and nested pipeline strings are deterministic data. Module and
  callback passes contain executable behavior, so each such step must be wrapped
  with an explicit stable version through `add_pass/3` or `versioned/2`. Function
  bodies, processes, references, and native handles are never hashed.

  `declaration/1` projects the executable plan to deterministic data, while
  `identity/1` hashes that projection. Both are computed from the current struct;
  no cached identity can drift from a modified plan.
  """

  alias Beaver.MLIR.CompilationRuntime.CacheKey
  alias Beaver.MLIR.Transform.FixedPoint
  alias Beaver.MLIR.Transform.Schedule, as: TransformSchedule

  @versioned_step :beaver_compilation_plan_versioned_step

  defstruct pipeline: [],
            transform_schedule: nil,
            transform_options: [],
            target: %{},
            schema_version: :static,
            desired_emit_version: nil,
            context_options: [],
            telemetry_metadata: %{}

  @type pass_step() ::
          binary()
          | module()
          | FixedPoint.t()
          | {binary(), [pass_step()]}
          | {binary(), binary(), function()}
          | {:beaver_compilation_plan_versioned_step, term(), term()}

  @type t() :: %__MODULE__{
          pipeline: [pass_step()],
          transform_schedule: TransformSchedule.Resolved.t() | binary() | nil,
          transform_options: keyword(),
          target: term(),
          schema_version: term(),
          desired_emit_version: integer() | nil,
          context_options: keyword(),
          telemetry_metadata: map()
        }

  @valid_options [
    :pipeline,
    :transform_schedule,
    :transform_options,
    :target,
    :schema_version,
    :desired_emit_version,
    :bytecode_version,
    :context_options,
    :telemetry_metadata
  ]

  @doc "Creates and validates a compilation plan without constructing native resources."
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    validate_keyword!(opts, @valid_options, "CompilationPlan options")

    desired_emit_version = desired_emit_version!(opts)
    transform_options = keyword_value!(opts, :transform_options, [])
    context_options = keyword_value!(opts, :context_options, [])
    telemetry_metadata = metadata_value!(opts)

    %__MODULE__{
      pipeline: opts |> Keyword.get(:pipeline, []) |> List.wrap() |> normalize_pipeline(),
      transform_schedule: Keyword.get(opts, :transform_schedule),
      transform_options: transform_options,
      target: Keyword.get(opts, :target, %{}),
      schema_version: Keyword.get(opts, :schema_version, :static),
      desired_emit_version: desired_emit_version,
      context_options: context_options,
      telemetry_metadata: telemetry_metadata
    }
    |> validate!()
  end

  @doc "Wraps one Composer pass payload with an explicit stable version."
  @spec versioned(term(), term()) :: pass_step()
  def versioned(pass, version) when not is_nil(version) do
    {@versioned_step, pass, version}
  end

  def versioned(_pass, nil) do
    raise ArgumentError, "a versioned pass requires a non-nil :version"
  end

  @doc "Appends one Composer-compatible pass, optionally with `:version`."
  @spec add_pass(t(), term(), keyword()) :: t()
  def add_pass(%__MODULE__{} = plan, pass, opts \\ []) do
    validate_keyword!(opts, [:version], "add_pass options")

    step =
      case Keyword.fetch(opts, :version) do
        {:ok, version} -> versioned(pass, version)
        :error -> normalize_pass_step(pass)
      end

    %{plan | pipeline: plan.pipeline ++ [step]}
    |> validate!()
  end

  @doc "Appends a nested Composer pass scope."
  @spec nested(t(), binary(), [pass_step()]) :: t()
  def nested(%__MODULE__{} = plan, operation_name, passes)
      when is_binary(operation_name) and is_list(passes) do
    %{plan | pipeline: plan.pipeline ++ [{operation_name, normalize_pipeline(passes)}]}
    |> validate!()
  end

  @doc "Sets the Transform schedule and its execution options."
  @spec set_transform_schedule(t(), TransformSchedule.Resolved.t() | binary() | nil, keyword()) ::
          t()
  def set_transform_schedule(%__MODULE__{} = plan, schedule, opts \\ []) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, ":transform_options must be a keyword list"
    end

    %{plan | transform_schedule: schedule, transform_options: opts}
    |> validate!()
  end

  @doc "Sets the target configuration included in the plan identity."
  @spec set_target(t(), term()) :: t()
  def set_target(%__MODULE__{} = plan, target), do: validate!(%{plan | target: target})

  @doc "Sets the dynamic dialect/schema identity."
  @spec set_schema_version(t(), term()) :: t()
  def set_schema_version(%__MODULE__{} = plan, version) do
    validate!(%{plan | schema_version: version})
  end

  @doc "Sets the desired MLIR bytecode emission version."
  @spec set_bytecode_version(t(), integer() | :current | nil) :: t()
  def set_bytecode_version(%__MODULE__{} = plan, version) do
    validate!(%{plan | desired_emit_version: validate_bytecode_version!(version)})
  end

  @doc "Sets options used when the compilation runtime creates an MLIR context."
  @spec set_context_options(t(), keyword()) :: t()
  def set_context_options(%__MODULE__{} = plan, opts) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, ":context_options must be a keyword list")

    validate!(%{plan | context_options: opts})
  end

  @doc "Sets deterministic metadata attached to compilation and artifact telemetry."
  @spec set_telemetry_metadata(t(), map() | keyword()) :: t()
  def set_telemetry_metadata(%__MODULE__{} = plan, metadata) do
    validate!(%{plan | telemetry_metadata: metadata_map!(metadata)})
  end

  @doc "Returns the deterministic, callback-free declaration represented by a plan."
  @spec declaration(t()) :: map()
  def declaration(%__MODULE__{} = plan) do
    validate_shape!(plan)

    declaration = %{
      pipeline: Enum.map(plan.pipeline, &pass_declaration!/1),
      transform_schedule: transform_schedule_declaration!(plan.transform_schedule),
      transform_options: transform_options_declaration!(plan),
      target: plan.target,
      schema_version: plan.schema_version,
      bytecode_version: plan.desired_emit_version || :current,
      context_options: canonical_keyword!(plan.context_options, :context_options),
      telemetry_metadata: plan.telemetry_metadata
    }

    ensure_deterministic!(declaration, "compilation plan declaration")
  end

  @doc "Returns the stable SHA-256 identity of `declaration/1`."
  @spec identity(t()) :: binary()
  def identity(%__MODULE__{} = plan) do
    CacheKey.lookup(%{compilation_plan: {1, declaration(plan)}})
  end

  @doc false
  @spec executable_pipeline(t()) :: list()
  def executable_pipeline(%__MODULE__{pipeline: pipeline}) do
    Enum.map(pipeline, &executable_pass/1)
  end

  @doc "Validates a plan and returns it unchanged."
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = plan) do
    _identity = identity(plan)
    plan
  end

  def validate!(other) do
    raise ArgumentError,
          "defcompiler must return a Beaver.MLIR.CompilationPlan, got: #{inspect(other, limit: 5)}"
  end

  @doc "Defines a zero-arity function that returns a validated compilation plan."
  defmacro defcompiler(call, opts_or_block) do
    plan_module = __MODULE__

    body =
      case opts_or_block do
        [do: block] ->
          block

        opts when is_list(opts) ->
          quote do
            unquote(plan_module).new(unquote(opts))
          end
      end

    quote do
      def unquote(call) do
        unquote(body)
        |> unquote(plan_module).validate!()
      end
    end
  end

  defp desired_emit_version!(opts) do
    desired = Keyword.fetch(opts, :desired_emit_version)
    alias_value = Keyword.fetch(opts, :bytecode_version)

    case {desired, alias_value} do
      {{:ok, left}, {:ok, right}} ->
        left = validate_bytecode_version!(left)
        right = validate_bytecode_version!(right)

        if left == right do
          left
        else
          raise ArgumentError, ":desired_emit_version and :bytecode_version disagree"
        end

      {{:ok, value}, _} ->
        validate_bytecode_version!(value)

      {_, {:ok, value}} ->
        validate_bytecode_version!(value)

      _ ->
        nil
    end
  end

  defp validate_bytecode_version!(:current), do: nil
  defp validate_bytecode_version!(value) when is_integer(value) or is_nil(value), do: value

  defp validate_bytecode_version!(value) do
    raise ArgumentError, ":desired_emit_version must be an integer or nil, got: #{inspect(value)}"
  end

  defp keyword_value!(opts, key, default) do
    value = Keyword.get(opts, key, default)

    unless Keyword.keyword?(value) do
      raise ArgumentError, "#{inspect(key)} must be a keyword list"
    end

    value
  end

  defp metadata_value!(opts) do
    opts |> Keyword.get(:telemetry_metadata, %{}) |> metadata_map!()
  end

  defp metadata_map!(metadata) when is_map(metadata), do: metadata

  defp metadata_map!(metadata) when is_list(metadata) do
    if Keyword.keyword?(metadata) do
      Map.new(metadata)
    else
      raise ArgumentError, ":telemetry_metadata must be a map or keyword list"
    end
  end

  defp metadata_map!(_metadata) do
    raise ArgumentError, ":telemetry_metadata must be a map or keyword list"
  end

  defp validate_keyword!(opts, allowed, label) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "#{label} must be a keyword list, got: #{inspect(opts)}"
    end

    keys = Keyword.keys(opts)

    if length(keys) != length(Enum.uniq(keys)) do
      raise ArgumentError, "#{label} must not contain duplicate keys"
    end

    case keys -- allowed do
      [] -> :ok
      unsupported -> raise ArgumentError, "unsupported #{label}: #{inspect(unsupported)}"
    end
  end

  defp normalize_pipeline(pipeline), do: Enum.map(pipeline, &normalize_pass_step/1)

  defp normalize_pass_step({operation_name, passes})
       when is_binary(operation_name) and is_list(passes) do
    {operation_name, normalize_pipeline(passes)}
  end

  defp normalize_pass_step({pass, [version: version]}), do: versioned(pass, version)
  defp normalize_pass_step({@versioned_step, pass, version}), do: versioned(pass, version)
  defp normalize_pass_step(pass), do: pass

  defp pass_declaration!({@versioned_step, pass, version}) do
    version = stable_version!(version)
    {:versioned, versioned_pass_declaration!(pass), version}
  end

  defp pass_declaration!(%FixedPoint{} = fixed_point) do
    {:fixed_point,
     %{
       name: fixed_point.name,
       pipeline: Enum.map(fixed_point.pipeline, &pass_declaration!/1),
       max_iterations: fixed_point.max_iterations,
       on_convergence_failure: fixed_point.on_convergence_failure
     }}
  end

  defp pass_declaration!({operation_name, passes})
       when is_binary(operation_name) and is_list(passes) do
    {:nested, operation_name, Enum.map(passes, &pass_declaration!/1)}
  end

  defp pass_declaration!(pipeline) when is_binary(pipeline), do: {:pipeline, pipeline}

  defp pass_declaration!(module) when is_atom(module) do
    raise ArgumentError,
          "module pass #{inspect(module)} requires an explicit :version"
  end

  defp pass_declaration!({_argument, _operation_name, callback}) when is_function(callback) do
    raise ArgumentError, "callback-backed pass requires an explicit :version"
  end

  defp pass_declaration!(callback) when is_function(callback) do
    raise ArgumentError,
          "callback-backed pass requires an explicit :version and a Composer pass tuple"
  end

  defp pass_declaration!(pass) do
    raise ArgumentError,
          "unsupported or non-deterministic Composer pass: #{inspect(pass, limit: 5)}"
  end

  defp versioned_pass_declaration!(module) when is_atom(module) do
    {:module, Atom.to_string(module)}
  end

  defp versioned_pass_declaration!({argument, operation_name, callback})
       when is_function(callback) do
    {:callback, ensure_deterministic!(argument, "pass argument"),
     ensure_deterministic!(operation_name, "pass operation")}
  end

  defp versioned_pass_declaration!({operation_name, passes})
       when is_binary(operation_name) and is_list(passes) do
    {:nested, operation_name, Enum.map(passes, &pass_declaration!/1)}
  end

  defp versioned_pass_declaration!(pipeline) when is_binary(pipeline),
    do: {:pipeline, pipeline}

  defp versioned_pass_declaration!(callback) when is_function(callback) do
    raise ArgumentError,
          "a callback Composer pass must be {argument, operation_name, callback}"
  end

  defp versioned_pass_declaration!(pass) do
    raise ArgumentError,
          "unsupported Composer pass payload: #{inspect(pass, limit: 5)}"
  end

  defp executable_pass({@versioned_step, pass, _version}), do: executable_pass(pass)

  defp executable_pass(%FixedPoint{} = fixed_point) do
    %{fixed_point | pipeline: Enum.map(fixed_point.pipeline, &executable_pass/1)}
  end

  defp executable_pass({operation_name, passes})
       when is_binary(operation_name) and is_list(passes) do
    {operation_name, Enum.map(passes, &executable_pass/1)}
  end

  defp executable_pass(pass), do: pass

  defp transform_schedule_declaration!(nil), do: :none

  defp transform_schedule_declaration!(%TransformSchedule.Resolved{} = schedule) do
    TransformSchedule.cache_identity(schedule)
  end

  defp transform_schedule_declaration!(schedule) when is_binary(schedule) do
    TransformSchedule.cache_identity(schedule)
  end

  defp transform_schedule_declaration!(schedule) do
    raise ArgumentError,
          ":transform_schedule must be resolved data or MLIR text/bytecode, got: #{inspect(schedule, limit: 5)}"
  end

  defp transform_options_declaration!(%__MODULE__{transform_schedule: nil}), do: :none

  defp transform_options_declaration!(%__MODULE__{transform_options: opts}) do
    opts |> canonical_keyword!(:transform_options) |> Map.new()
  end

  defp canonical_keyword!(opts, label) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "#{inspect(label)} must be a keyword list")

    keys = Keyword.keys(opts)

    if length(keys) != length(Enum.uniq(keys)) do
      raise ArgumentError, "#{inspect(label)} must not contain duplicate keys"
    end

    Enum.sort(opts)
  end

  defp validate_shape!(%__MODULE__{} = plan) do
    unless is_list(plan.pipeline), do: raise(ArgumentError, ":pipeline must be a list")

    unless Keyword.keyword?(plan.transform_options),
      do: raise(ArgumentError, ":transform_options must be a keyword list")

    unless Keyword.keyword?(plan.context_options),
      do: raise(ArgumentError, ":context_options must be a keyword list")

    unless is_map(plan.telemetry_metadata),
      do: raise(ArgumentError, ":telemetry_metadata must be a map")

    unless is_integer(plan.desired_emit_version) or is_nil(plan.desired_emit_version) do
      raise ArgumentError, ":desired_emit_version must be an integer or nil"
    end

    :ok
  end

  defp ensure_deterministic!(value, label) do
    _digest = CacheKey.lookup(%{compilation_plan_value: value})
    value
  rescue
    error in ArgumentError ->
      reraise ArgumentError,
              [message: "#{label} must contain only deterministic data: #{error.message}"],
              __STACKTRACE__
  end

  defp stable_version!(nil),
    do: raise(ArgumentError, "a versioned pass requires a non-nil :version")

  defp stable_version!(version), do: ensure_deterministic!(version, "pass version")
end
