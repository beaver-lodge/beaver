alias Beaver.MLIR.{ConversionTarget, RewritePatternSet, Type, TypeConverter}

defmodule Beaver.MLIR.Conversion.Plan do
  @moduledoc """
  Reusable declarations for dialect conversion: legality rules, type conversions,
  materializations and patterns. Plans hold no native handles and work across contexts.
  Each run creates fresh resources and destroys patterns before converters and targets.

  `declaration/1` omits callback bodies and runtime state. Supply callback `:version`
  metadata for reproducible declarations; absent versions are `:unversioned`.
  """

  alias Beaver.{MLIR, Native}
  alias Beaver.MLIR.Conversion.Kernel, as: ConversionKernel
  alias Beaver.Pattern.Native.Descriptor

  defstruct mode: :full, timeout: 30_000, folding_mode: nil, build_materializations: nil, entries: []

  @type mode() :: :full | :partial
  @type option() ::
          {:mode, mode()}
          | {:timeout, non_neg_integer() | nil}
          | {:folding_mode, :never | :before_patterns | :after_patterns | nil}
          | {:build_materializations, boolean() | nil}

  @type t() :: %__MODULE__{
          mode: mode(),
          timeout: non_neg_integer() | nil,
          folding_mode: :never | :before_patterns | :after_patterns | nil,
          build_materializations: boolean() | nil,
          entries: [term()]
        }

  @doc """
  Creates a plan with `:mode` (`:full` or `:partial`), `:timeout` (default 30_000),
  `:folding_mode` (`:never`, `:before_patterns`, `:after_patterns` or nil) and
  `:build_materializations` (boolean or nil).
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) when is_list(opts) do
    validate_plan_options!(opts)

    validators = %{
      mode: &validate_mode!/1,
      timeout: &validate_plan_timeout!/1,
      folding_mode: &validate_folding_mode!/1,
      build_materializations: &validate_build_materializations!/1
    }

    struct!(
      __MODULE__,
      Enum.map(Enum.uniq_by(opts, &elem(&1, 0)), fn {key, value} -> {key, validators[key].(value)} end)
    )
  end

  for {name, field} <- [
        add_legal_op: :op,
        add_illegal_op: :op,
        add_legal_dialect: :dialect,
        add_illegal_dialect: :dialect
      ] do
    @spec unquote(name)(t(), String.Chars.t()) :: t()
    def unquote(name)(%__MODULE__{} = plan, name),
      do: append_entry(plan, {unquote(name), to_string(name)})

    defp entry_declaration({unquote(name), name}), do: %{unquote(field) => name, kind: unquote(name)}
  end

  for {name, field} <- [add_dynamically_legal_op: :op, add_dynamically_legal_dialect: :dialect] do
    @spec unquote(name)(t(), String.Chars.t(), ConversionTarget.legality_callback(), keyword()) :: t()
    def unquote(name)(%__MODULE__{} = plan, name, callback, opts \\ []) when is_function(callback, 1) do
      opts = validate_callback_opts!(opts, [:version], unquote(Atom.to_string(name)))
      append_entry(plan, {unquote(name), to_string(name), callback, opts})
    end

    defp entry_declaration({unquote(name), name, _callback, opts}),
      do: %{unquote(field) => name, kind: unquote(name), version: callback_version(opts)}
  end

  @spec mark_recursively_legal(t(), String.Chars.t(), ConversionTarget.legality_callback() | nil | keyword(), keyword()) ::
          t()
  def mark_recursively_legal(plan, name, callback_or_opts \\ nil, opts \\ [])

  def mark_recursively_legal(%__MODULE__{} = plan, name, opts, []) when is_list(opts) do
    opts = validate_callback_opts!(opts, [:version], "mark_recursively_legal")
    append_entry(plan, {:mark_recursively_legal, to_string(name), nil, opts})
  end

  def mark_recursively_legal(%__MODULE__{} = plan, name, callback, opts)
      when is_nil(callback) or is_function(callback, 1) do
    opts = validate_callback_opts!(opts, [:version], "mark_recursively_legal")
    append_entry(plan, {:mark_recursively_legal, to_string(name), callback, opts})
  end

  for {name, arity, callback_type} <- [
        {:mark_unknown_dynamically_legal, 1, quote(do: ConversionTarget.legality_callback())},
        {:add_conversion, 1, quote(do: (Type.t() -> TypeConverter.conversion_result()))},
        {:add_1_to_n_conversion, 1, quote(do: (Type.t() -> TypeConverter.one_to_n_result()))},
        {:add_source_materialization, 4, quote(do: function())},
        {:add_target_materialization, 5, quote(do: function())},
        {:add_1_to_n_target_materialization, 5, quote(do: function())}
      ] do
    @spec unquote(name)(t(), unquote(callback_type), keyword()) :: t()
    def unquote(name)(%__MODULE__{} = plan, callback, opts \\ []) when is_function(callback, unquote(arity)) do
      opts = validate_callback_opts!(opts, [:version], unquote(Atom.to_string(name)))
      append_entry(plan, {unquote(name), callback, opts})
    end
  end

  @doc """
  Adds a native type mapping. Assembly strings resolve in each run's context;
  the mapping has no callbacks and needs no version metadata.
  """
  @spec add_conversion_map(t(), [String.t()], String.t()) :: t()
  def add_conversion_map(%__MODULE__{} = plan, source_types, target_type)
      when is_list(source_types) and is_binary(target_type) do
    unless source_types != [] and Enum.all?(source_types, &is_binary/1),
      do: raise(ArgumentError, "conversion map source types must be a non-empty list of strings")

    append_entry(plan, {:add_conversion_map, source_types, target_type})
  end

  @spec add_conversion_pattern(t(), String.Chars.t(), MLIR.ConversionPattern.callback(), keyword()) :: t()
  def add_conversion_pattern(%__MODULE__{} = plan, root_name, callback, opts \\ [])
      when is_function(callback, 3) do
    opts =
      validate_callback_opts!(opts, [:version, :benefit, :one_to_n, :timeout], "add_conversion_pattern")

    benefit = Keyword.get(opts, :benefit, 1)

    unless is_integer(benefit) and benefit >= 0,
      do: raise(ArgumentError, ":benefit must be a non-negative integer, got: #{inspect(benefit)}")

    one_to_n = Keyword.get(opts, :one_to_n, false)

    unless is_boolean(one_to_n), do: raise(ArgumentError, ":one_to_n must be boolean, got: #{inspect(one_to_n)}")

    timeout = Keyword.get(opts, :timeout)

    if timeout != nil and not (is_integer(timeout) and timeout >= 0),
      do: raise(ArgumentError, ":timeout must be a non-negative integer or nil, got: #{inspect(timeout)}")

    append_entry(plan, {:add_conversion_pattern, to_string(root_name), callback, opts})
  end

  @spec add_pattern(t(), Descriptor.t(), keyword()) :: t()
  def add_pattern(%__MODULE__{} = plan, %Descriptor{} = descriptor, opts \\ []) do
    opts = validate_callback_opts!(opts, [:version], "add_pattern")
    append_entry(plan, {:add_pattern, descriptor, opts})
  end

  @doc """
  Runs once per materialization with the mutable pattern set and converter;
  must return `:ok`. Native patterns installed here avoid per-match BEAM callbacks.
  """
  @spec add_pattern_population(t(), (RewritePatternSet.t(), TypeConverter.t() -> :ok), keyword()) ::
          t()
  def add_pattern_population(%__MODULE__{} = plan, callback, opts \\ [])
      when is_function(callback, 2) do
    opts = validate_callback_opts!(opts, [:version], "add_pattern_population")
    append_entry(plan, {:add_pattern_population, callback, opts})
  end

  @doc """
  Verifies and loads a compiler kernel during materialization. Its native patterns
  avoid BEAM callbacks. `artifact_path` must be absolute; `:expected` constrains
  provider schema, runtime ABI, target and capabilities.
  """
  @spec add_external_pattern_population(t(), ConversionKernel.Manifest.t(), Path.t(), keyword()) ::
          t()
  def add_external_pattern_population(
        %__MODULE__{} = plan,
        %ConversionKernel.Manifest{} = manifest,
        artifact_path,
        opts \\ []
      ) do
    opts =
      validate_callback_opts!(opts, [:expected], "add_external_pattern_population")

    ConversionKernel.Manifest.validate!(manifest)
    artifact_path = ConversionKernel.validate_artifact_path!(artifact_path)
    append_entry(plan, {:add_external_pattern_population, manifest, artifact_path, opts})
  end

  @doc """
  Returns configuration and declaration metadata, omitting callback bodies and
  runtime state. Reproducibility requires explicit callback `:version` metadata.
  """
  @spec declaration(t()) :: map()
  def declaration(%__MODULE__{} = plan),
    do: plan |> Map.from_struct() |> Map.update!(:entries, &Enum.map(&1, fn entry -> entry_declaration(entry) end))

  @doc """
  Executes on a module or operation with fresh, scoped native resources.
  Returns `MLIR.Conversion.result()`.
  """
  @spec run(t(), MLIR.Conversion.conversion_ir()) :: MLIR.Conversion.result()
  def run(%__MODULE__{} = plan, ir), do: execute(plan, ir, false)

  @doc """
  Returns `{run_result, receipt}` with bounded conversion callback profiling.
  """
  @spec profile(t(), MLIR.Conversion.conversion_ir()) ::
          {MLIR.Conversion.result(), MLIR.Conversion.Profile.receipt()}
  def profile(%__MODULE__{} = plan, ir), do: execute(plan, ir, true)

  defp execute(plan, ir, profile?) do
    context = MLIR.context(ir)

    target_opts = if plan.timeout, do: [timeout: plan.timeout], else: []

    opts =
      Map.take(plan, [:timeout, :folding_mode, :build_materializations]) |> Enum.reject(fn {_, v} -> is_nil(v) end)

    function = if profile?, do: :profile, else: :apply

    Native.with_resource(ConversionTarget.create(context, target_opts), &ConversionTarget.destroy/1, fn target ->
      TypeConverter.with(target_opts, fn converter ->
        patterns = RewritePatternSet.create(context)
        populate_or_destroy!(plan.entries, context, target, converter, patterns, plan.timeout)

        # Conversion owns the frozen set once submitted, including worker failure.
        apply(MLIR.Conversion, function, [plan.mode, ir, target, patterns, opts])
      end)
    end)
  end

  @doc """
  Executes the conversion plan on the given IR, returning the converted IR or raising `MLIR.Conversion.Error`.
  """
  @spec run!(t(), MLIR.Conversion.conversion_ir()) :: MLIR.Conversion.conversion_ir()
  def run!(%__MODULE__{} = plan, ir) do
    case run(plan, ir) do
      {:ok, converted, _diagnostics} -> converted
      {:error, %MLIR.Conversion.Error{} = error} -> raise error
    end
  end

  @doc "Executes a profiled conversion plan, returning the IR and receipt or raising on failure."
  @spec profile!(t(), MLIR.Conversion.conversion_ir()) ::
          {MLIR.Conversion.conversion_ir(), MLIR.Conversion.Profile.receipt()}
  def profile!(%__MODULE__{} = plan, ir) do
    case profile(plan, ir) do
      {{:ok, converted, _diagnostics}, receipt} -> {converted, receipt}
      {{:error, %MLIR.Conversion.Error{} = error}, _receipt} -> raise error
    end
  end

  defp append_entry(%__MODULE__{entries: entries} = plan, entry), do: %{plan | entries: entries ++ [entry]}

  defp validate_plan_options!(opts),
    do: validate_callback_opts!(opts, [:mode, :timeout, :folding_mode, :build_materializations], "Plan")

  defp validate_mode!(mode) when mode in [:full, :partial], do: mode

  defp validate_mode!(mode), do: raise(ArgumentError, ":mode must be :full or :partial, got: #{inspect(mode)}")

  defp validate_plan_timeout!(nil), do: nil
  defp validate_plan_timeout!(timeout) when is_integer(timeout) and timeout >= 0, do: timeout

  defp validate_plan_timeout!(timeout),
    do: raise(ArgumentError, ":timeout must be a non-negative integer or nil, got: #{inspect(timeout)}")

  defp validate_folding_mode!(mode) when mode in [nil, :never, :before_patterns, :after_patterns], do: mode

  defp validate_folding_mode!(mode), do: raise(ArgumentError, "unsupported conversion folding mode: #{inspect(mode)}")

  defp validate_build_materializations!(value) when is_boolean(value) or is_nil(value), do: value

  defp validate_build_materializations!(value),
    do: raise(ArgumentError, "build_materializations must be boolean or nil, got: #{inspect(value)}")

  defp validate_callback_opts!(opts, allowed, label) do
    unless Keyword.keyword?(opts),
      do: raise(ArgumentError, "#{label} options must be a keyword list, got: #{inspect(opts)}")

    case Keyword.keys(opts) -- allowed do
      [] -> opts
      unsupported -> raise ArgumentError, "unsupported #{label} options: #{inspect(unsupported)}"
    end
  end

  defp entry_declaration({:mark_recursively_legal, name, cb, opts}) do
    base = %{kind: :mark_recursively_legal, op: name}

    if cb != nil, do: Map.put(base, :version, callback_version(opts)), else: base
  end

  defp entry_declaration({kind, _callback, opts})
       when kind in ~w(
         mark_unknown_dynamically_legal add_conversion add_1_to_n_conversion add_source_materialization
         add_target_materialization add_1_to_n_target_materialization add_pattern_population
       )a,
       do: %{kind: kind, version: callback_version(opts)}

  defp entry_declaration({:add_conversion_map, source_types, target_type}),
    do: %{kind: :add_conversion_map, source_types: source_types, target_type: target_type}

  defp entry_declaration({:add_conversion_pattern, root, _cb, opts}),
    do: %{
      kind: :add_conversion_pattern,
      root: root,
      benefit: Keyword.get(opts, :benefit, 1),
      one_to_n: Keyword.get(opts, :one_to_n, false),
      timeout: Keyword.get(opts, :timeout),
      version: callback_version(opts)
    }

  defp entry_declaration({:add_pattern, descriptor, opts}),
    do: %{
      kind: :add_pattern,
      name: descriptor.name,
      root: descriptor.root,
      benefit: descriptor.benefit,
      version: callback_version(opts)
    }

  defp entry_declaration({:add_external_pattern_population, manifest, _path, _opts}),
    do: %{
      kind: :add_external_pattern_population,
      provider: manifest.provider,
      manifest_digest: ConversionKernel.Manifest.digest(manifest),
      identity_digest: ConversionKernel.Manifest.identity_digest(manifest),
      artifact_sha256: manifest.artifact_sha256,
      patterns: manifest.patterns,
      bootstrap: manifest.bootstrap
    }

  defp callback_version(opts), do: Keyword.get(opts, :version, :unversioned)

  defp populate_or_destroy!(entries, ctx, target, converter, patterns, timeout) do
    resources = %{context: ctx, target: target, converter: converter, patterns: patterns, timeout: timeout}
    Enum.each(entries, &populate_entry(&1, resources))
  catch
    kind, reason ->
      RewritePatternSet.threaded_destroy(ctx, patterns)
      :erlang.raise(kind, reason, __STACKTRACE__)
  end

  defp populate_entry({kind, name}, %{target: target})
       when kind in [:add_legal_op, :add_illegal_op, :add_legal_dialect, :add_illegal_dialect],
       do: apply(ConversionTarget, kind, [target, name])

  defp populate_entry({kind, name, callback, _opts}, %{target: target})
       when kind in [:add_dynamically_legal_op, :add_dynamically_legal_dialect, :mark_recursively_legal],
       do: apply(ConversionTarget, kind, [target, name, callback])

  defp populate_entry({:mark_unknown_dynamically_legal, callback, _opts}, %{target: target}),
    do: ConversionTarget.mark_unknown_dynamically_legal(target, callback)

  defp populate_entry({kind, callback, _opts}, %{converter: converter})
       when kind in ~w(
         add_conversion add_1_to_n_conversion add_source_materialization add_target_materialization
         add_1_to_n_target_materialization
       )a,
       do: apply(TypeConverter, kind, [converter, callback])

  defp populate_entry({:add_conversion_map, source_types, target_type}, resources) do
    sources = Enum.map(source_types, &Type.get(&1, ctx: resources.context))
    target = Type.get(target_type, ctx: resources.context)
    TypeConverter.add_conversion_map(resources.converter, sources, target)
  end

  defp populate_entry({:add_conversion_pattern, root_name, callback, opts}, resources) do
    opts = conversion_pattern_opts(opts, resources.context, resources.timeout)
    MLIR.ConversionPattern.add(resources.patterns, root_name, resources.converter, callback, opts)
  end

  defp populate_entry({:add_pattern, descriptor, _opts}, resources),
    do: RewritePatternSet.add(resources.patterns, descriptor, ctx: resources.context)

  defp populate_entry({:add_pattern_population, callback, _opts}, resources) do
    case callback.(resources.patterns, resources.converter) do
      :ok -> :ok
      other -> raise ArgumentError, "pattern population must return :ok, got: #{inspect(other)}"
    end
  end

  defp populate_entry({:add_external_pattern_population, manifest, artifact_path, opts}, resources) do
    ConversionKernel.populate!(resources.patterns, resources.converter, manifest, artifact_path,
      expected: Keyword.get(opts, :expected, [])
    )
  end

  defp conversion_pattern_opts(opts, ctx, plan_timeout) do
    pattern_opts =
      opts
      |> Keyword.take([:benefit, :one_to_n, :timeout])
      |> Keyword.reject(fn {key, value} -> key == :timeout and is_nil(value) end)
      |> Keyword.put(:ctx, ctx)

    if plan_timeout && not Keyword.has_key?(pattern_opts, :timeout),
      do: Keyword.put(pattern_opts, :timeout, plan_timeout),
      else: pattern_opts
  end
end
