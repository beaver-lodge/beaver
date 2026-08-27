defmodule Beaver.MLIR.Conversion.Plan do
  @moduledoc """
  An inspectable, declarative, scoped composition layer for MLIR dialect conversion.

  A conversion plan records target legality rules, type conversions, materializations,
  and rewrite patterns into a reusable data structure. When executed via `run/2`
  or `run!/2`, the plan materializes fresh conversion targets, type converters,
  and pattern sets in declaration order for the target `MLIR.Context`.

  ## Callbacks and Metadata

  Callbacks registered with plan builders accept optional `:version` metadata.
  `declaration/1` returns a deterministic map of plan configuration and step
  metadata with function bodies and runtime state omitted. Unversioned callbacks
  are marked as `:unversioned`. This metadata is only reproducible across runs
  when callback versions are explicitly provided.

  ## Ownership & Scoping

  Plans do not hold native handles; they can be safely reused across multiple
  MLIR contexts. During `run/2`, temporary native resources (`MLIR.ConversionTarget`,
  `MLIR.TypeConverter`, `MLIR.RewritePatternSet`) are created and cleaned up.
  In case of errors or caller termination, resources are cleaned up deterministically:
  pattern destruction occurs before type converter destruction, which occurs before
  conversion target destruction.
  """

  alias Beaver.MLIR
  alias Beaver.Pattern.Native.Descriptor

  defstruct mode: :full,
            timeout: 30_000,
            folding_mode: nil,
            build_materializations: nil,
            entries: []

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
  Creates a new conversion plan.

  Options:
    * `:mode` - `:full` (default) or `:partial`.
    * `:timeout` - Timeout in milliseconds for conversion and callbacks (default `30_000`).
    * `:folding_mode` - `:never`, `:before_patterns`, `:after_patterns`, or `nil`.
    * `:build_materializations` - boolean or `nil`.
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) when is_list(opts) do
    validate_plan_options!(opts)

    mode = opts |> Keyword.get(:mode, :full) |> validate_mode!()
    timeout = opts |> Keyword.get(:timeout, 30_000) |> validate_plan_timeout!()
    folding_mode = opts |> Keyword.get(:folding_mode) |> validate_folding_mode!()

    build_materializations =
      opts
      |> Keyword.get(:build_materializations)
      |> validate_build_materializations!()

    %__MODULE__{
      mode: mode,
      timeout: timeout,
      folding_mode: folding_mode,
      build_materializations: build_materializations,
      entries: []
    }
  end

  @spec add_legal_op(t(), String.Chars.t()) :: t()
  def add_legal_op(%__MODULE__{} = plan, name) do
    append_entry(plan, {:add_legal_op, to_string(name)})
  end

  @spec add_illegal_op(t(), String.Chars.t()) :: t()
  def add_illegal_op(%__MODULE__{} = plan, name) do
    append_entry(plan, {:add_illegal_op, to_string(name)})
  end

  @spec add_legal_dialect(t(), String.Chars.t()) :: t()
  def add_legal_dialect(%__MODULE__{} = plan, name) do
    append_entry(plan, {:add_legal_dialect, to_string(name)})
  end

  @spec add_illegal_dialect(t(), String.Chars.t()) :: t()
  def add_illegal_dialect(%__MODULE__{} = plan, name) do
    append_entry(plan, {:add_illegal_dialect, to_string(name)})
  end

  @spec add_dynamically_legal_op(
          t(),
          String.Chars.t(),
          MLIR.ConversionTarget.legality_callback(),
          keyword()
        ) :: t()
  def add_dynamically_legal_op(%__MODULE__{} = plan, name, callback, opts \\ [])
      when is_function(callback, 1) do
    opts = validate_callback_opts!(opts, [:version], "add_dynamically_legal_op")
    append_entry(plan, {:add_dynamically_legal_op, to_string(name), callback, opts})
  end

  @spec add_dynamically_legal_dialect(
          t(),
          String.Chars.t(),
          MLIR.ConversionTarget.legality_callback(),
          keyword()
        ) :: t()
  def add_dynamically_legal_dialect(%__MODULE__{} = plan, name, callback, opts \\ [])
      when is_function(callback, 1) do
    opts = validate_callback_opts!(opts, [:version], "add_dynamically_legal_dialect")
    append_entry(plan, {:add_dynamically_legal_dialect, to_string(name), callback, opts})
  end

  @spec mark_recursively_legal(
          t(),
          String.Chars.t(),
          MLIR.ConversionTarget.legality_callback() | nil | keyword(),
          keyword()
        ) :: t()
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

  @spec mark_unknown_dynamically_legal(
          t(),
          MLIR.ConversionTarget.legality_callback(),
          keyword()
        ) :: t()
  def mark_unknown_dynamically_legal(%__MODULE__{} = plan, callback, opts \\ [])
      when is_function(callback, 1) do
    opts = validate_callback_opts!(opts, [:version], "mark_unknown_dynamically_legal")
    append_entry(plan, {:mark_unknown_dynamically_legal, callback, opts})
  end

  @spec add_conversion(
          t(),
          (MLIR.Type.t() -> MLIR.TypeConverter.conversion_result()),
          keyword()
        ) :: t()
  def add_conversion(%__MODULE__{} = plan, callback, opts \\ [])
      when is_function(callback, 1) do
    opts = validate_callback_opts!(opts, [:version], "add_conversion")
    append_entry(plan, {:add_conversion, callback, opts})
  end

  @doc """
  Adds a deterministic native type mapping to the plan.

  Type assembly strings are resolved in the plan's context during
  materialization. Unlike `add_conversion/3`, this entry contains no
  executable callback and is cache-stable without a version annotation.
  """
  @spec add_conversion_map(t(), [String.t()], String.t()) :: t()
  def add_conversion_map(%__MODULE__{} = plan, source_types, target_type)
      when is_list(source_types) and is_binary(target_type) do
    unless source_types != [] and Enum.all?(source_types, &is_binary/1) do
      raise ArgumentError, "conversion map source types must be a non-empty list of strings"
    end

    append_entry(plan, {:add_conversion_map, source_types, target_type})
  end

  @spec add_1_to_n_conversion(
          t(),
          (MLIR.Type.t() -> MLIR.TypeConverter.one_to_n_result()),
          keyword()
        ) :: t()
  def add_1_to_n_conversion(%__MODULE__{} = plan, callback, opts \\ [])
      when is_function(callback, 1) do
    opts = validate_callback_opts!(opts, [:version], "add_1_to_n_conversion")
    append_entry(plan, {:add_1_to_n_conversion, callback, opts})
  end

  @spec add_source_materialization(t(), function(), keyword()) :: t()
  def add_source_materialization(%__MODULE__{} = plan, callback, opts \\ [])
      when is_function(callback, 4) do
    opts = validate_callback_opts!(opts, [:version], "add_source_materialization")
    append_entry(plan, {:add_source_materialization, callback, opts})
  end

  @spec add_target_materialization(t(), function(), keyword()) :: t()
  def add_target_materialization(%__MODULE__{} = plan, callback, opts \\ [])
      when is_function(callback, 5) do
    opts = validate_callback_opts!(opts, [:version], "add_target_materialization")
    append_entry(plan, {:add_target_materialization, callback, opts})
  end

  @spec add_1_to_n_target_materialization(t(), function(), keyword()) :: t()
  def add_1_to_n_target_materialization(%__MODULE__{} = plan, callback, opts \\ [])
      when is_function(callback, 5) do
    opts = validate_callback_opts!(opts, [:version], "add_1_to_n_target_materialization")
    append_entry(plan, {:add_1_to_n_target_materialization, callback, opts})
  end

  @spec add_conversion_pattern(
          t(),
          String.Chars.t(),
          MLIR.ConversionPattern.callback(),
          keyword()
        ) :: t()
  def add_conversion_pattern(%__MODULE__{} = plan, root_name, callback, opts \\ [])
      when is_function(callback, 3) do
    opts =
      validate_callback_opts!(
        opts,
        [:version, :benefit, :one_to_n, :timeout],
        "add_conversion_pattern"
      )

    benefit = Keyword.get(opts, :benefit, 1)

    unless is_integer(benefit) and benefit >= 0 do
      raise ArgumentError, ":benefit must be a non-negative integer, got: #{inspect(benefit)}"
    end

    one_to_n = Keyword.get(opts, :one_to_n, false)

    unless is_boolean(one_to_n) do
      raise ArgumentError, ":one_to_n must be boolean, got: #{inspect(one_to_n)}"
    end

    timeout = Keyword.get(opts, :timeout)

    if timeout != nil and not (is_integer(timeout) and timeout >= 0) do
      raise ArgumentError,
            ":timeout must be a non-negative integer or nil, got: #{inspect(timeout)}"
    end

    append_entry(plan, {:add_conversion_pattern, to_string(root_name), callback, opts})
  end

  @spec add_pattern(t(), Descriptor.t(), keyword()) :: t()
  def add_pattern(%__MODULE__{} = plan, %Descriptor{} = descriptor, opts \\ []) do
    opts = validate_callback_opts!(opts, [:version], "add_pattern")
    append_entry(plan, {:add_pattern, descriptor, opts})
  end

  @doc """
  Adds a pattern-set population callback to the plan.

  The callback runs once while a fresh plan is materialized and receives the
  mutable rewrite pattern set and its type converter. It must return `:ok`.
  This is the composition point for callback-free native pattern groups: the
  callback only installs patterns and is not invoked by the conversion worker.
  """
  @spec add_pattern_population(
          t(),
          (MLIR.RewritePatternSet.t(), MLIR.TypeConverter.t() -> :ok),
          keyword()
        ) :: t()
  def add_pattern_population(%__MODULE__{} = plan, callback, opts \\ [])
      when is_function(callback, 2) do
    opts = validate_callback_opts!(opts, [:version], "add_pattern_population")
    append_entry(plan, {:add_pattern_population, callback, opts})
  end

  @doc """
  Returns deterministic metadata for the given plan.

  Function bodies and runtime state are omitted. Callback entries include their
  `:version` if explicitly provided, or `:unversioned` if omitted.

  > Note: Declaration metadata is only deterministic and reproducible across processes
  > or runs when all callback versions are explicitly specified.
  """
  @spec declaration(t()) :: map()
  def declaration(%__MODULE__{} = plan) do
    %{
      mode: plan.mode,
      timeout: plan.timeout,
      folding_mode: plan.folding_mode,
      build_materializations: plan.build_materializations,
      entries: Enum.map(plan.entries, &entry_declaration/1)
    }
  end

  @doc """
  Executes the conversion plan on the given IR (`MLIR.Module` or `MLIR.Operation`).

  Fresh `MLIR.ConversionTarget`, `MLIR.TypeConverter`, and `MLIR.RewritePatternSet`
  instances are created for the duration of the conversion and cleaned up afterwards.
  Returns `MLIR.Conversion.result()`.
  """
  @spec run(t(), MLIR.Conversion.conversion_ir()) :: MLIR.Conversion.result()
  def run(%__MODULE__{} = plan, ir) do
    execute(plan, ir, false)
  end

  @doc """
  Executes the plan with bounded conversion callback profiling.

  The first tuple element is identical to `run/2`; the second is a
  `Beaver.MLIR.Conversion.Profile` receipt.
  """
  @spec profile(t(), MLIR.Conversion.conversion_ir()) ::
          {MLIR.Conversion.result(), MLIR.Conversion.Profile.receipt()}
  def profile(%__MODULE__{} = plan, ir) do
    execute(plan, ir, true)
  end

  defp execute(plan, ir, profile?) do
    context = MLIR.context(ir)

    target_opts = if plan.timeout, do: [timeout: plan.timeout], else: []
    target = MLIR.ConversionTarget.create(context, target_opts)

    try do
      converter_opts = if plan.timeout, do: [timeout: plan.timeout], else: []
      converter = MLIR.TypeConverter.create(converter_opts)

      try do
        patterns = MLIR.RewritePatternSet.create(context)

        populate_or_destroy!(
          plan.entries,
          context,
          target,
          converter,
          patterns,
          plan.timeout
        )

        conversion_opts =
          [
            timeout: plan.timeout,
            folding_mode: plan.folding_mode,
            build_materializations: plan.build_materializations
          ]
          |> Enum.reject(fn {_k, v} -> is_nil(v) end)

        # Conversion.apply/5 freezes the mutable set and transfers its ownership
        # to the native worker. Do not rescue around this call and try to destroy
        # the stale mutable handle: apply/5 already releases the owned frozen set
        # if starting the worker fails, and the worker releases it on every exit.
        if profile? do
          MLIR.Conversion.profile(plan.mode, ir, target, patterns, conversion_opts)
        else
          MLIR.Conversion.apply(plan.mode, ir, target, patterns, conversion_opts)
        end
      after
        MLIR.TypeConverter.destroy(converter)
      end
    after
      MLIR.ConversionTarget.destroy(target)
    end
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

  defp append_entry(%__MODULE__{entries: entries} = plan, entry) do
    %{plan | entries: entries ++ [entry]}
  end

  defp validate_plan_options!(opts) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "Plan options must be a keyword list, got: #{inspect(opts)}"
    end

    case Keyword.keys(opts) -- [:mode, :timeout, :folding_mode, :build_materializations] do
      [] -> :ok
      unsupported -> raise ArgumentError, "unsupported Plan options: #{inspect(unsupported)}"
    end
  end

  defp validate_mode!(mode) when mode in [:full, :partial], do: mode

  defp validate_mode!(mode) do
    raise ArgumentError, ":mode must be :full or :partial, got: #{inspect(mode)}"
  end

  defp validate_plan_timeout!(nil), do: nil
  defp validate_plan_timeout!(timeout) when is_integer(timeout) and timeout >= 0, do: timeout

  defp validate_plan_timeout!(timeout) do
    raise ArgumentError,
          ":timeout must be a non-negative integer or nil, got: #{inspect(timeout)}"
  end

  defp validate_folding_mode!(mode)
       when mode in [nil, :never, :before_patterns, :after_patterns],
       do: mode

  defp validate_folding_mode!(mode) do
    raise ArgumentError, "unsupported conversion folding mode: #{inspect(mode)}"
  end

  defp validate_build_materializations!(value) when is_boolean(value) or is_nil(value), do: value

  defp validate_build_materializations!(value) do
    raise ArgumentError,
          "build_materializations must be boolean or nil, got: #{inspect(value)}"
  end

  defp validate_callback_opts!(opts, allowed, label) do
    unless Keyword.keyword?(opts) do
      raise ArgumentError, "#{label} options must be a keyword list, got: #{inspect(opts)}"
    end

    case Keyword.keys(opts) -- allowed do
      [] ->
        opts

      unsupported ->
        raise ArgumentError, "unsupported #{label} options: #{inspect(unsupported)}"
    end
  end

  defp entry_declaration({:add_legal_op, name}), do: %{kind: :add_legal_op, op: name}
  defp entry_declaration({:add_illegal_op, name}), do: %{kind: :add_illegal_op, op: name}

  defp entry_declaration({:add_legal_dialect, name}),
    do: %{kind: :add_legal_dialect, dialect: name}

  defp entry_declaration({:add_illegal_dialect, name}),
    do: %{kind: :add_illegal_dialect, dialect: name}

  defp entry_declaration({:add_dynamically_legal_op, name, _cb, opts}) do
    %{kind: :add_dynamically_legal_op, op: name, version: callback_version(opts)}
  end

  defp entry_declaration({:add_dynamically_legal_dialect, name, _cb, opts}) do
    %{kind: :add_dynamically_legal_dialect, dialect: name, version: callback_version(opts)}
  end

  defp entry_declaration({:mark_recursively_legal, name, cb, opts}) do
    base = %{kind: :mark_recursively_legal, op: name}

    if cb != nil do
      Map.put(base, :version, callback_version(opts))
    else
      base
    end
  end

  defp entry_declaration({:mark_unknown_dynamically_legal, _cb, opts}) do
    %{kind: :mark_unknown_dynamically_legal, version: callback_version(opts)}
  end

  defp entry_declaration({:add_conversion, _cb, opts}) do
    %{kind: :add_conversion, version: callback_version(opts)}
  end

  defp entry_declaration({:add_conversion_map, source_types, target_type}) do
    %{kind: :add_conversion_map, source_types: source_types, target_type: target_type}
  end

  defp entry_declaration({:add_1_to_n_conversion, _cb, opts}) do
    %{kind: :add_1_to_n_conversion, version: callback_version(opts)}
  end

  defp entry_declaration({:add_source_materialization, _cb, opts}) do
    %{kind: :add_source_materialization, version: callback_version(opts)}
  end

  defp entry_declaration({:add_target_materialization, _cb, opts}) do
    %{kind: :add_target_materialization, version: callback_version(opts)}
  end

  defp entry_declaration({:add_1_to_n_target_materialization, _cb, opts}) do
    %{kind: :add_1_to_n_target_materialization, version: callback_version(opts)}
  end

  defp entry_declaration({:add_conversion_pattern, root, _cb, opts}) do
    %{
      kind: :add_conversion_pattern,
      root: root,
      benefit: Keyword.get(opts, :benefit, 1),
      one_to_n: Keyword.get(opts, :one_to_n, false),
      timeout: Keyword.get(opts, :timeout),
      version: callback_version(opts)
    }
  end

  defp entry_declaration({:add_pattern, descriptor, opts}) do
    %{
      kind: :add_pattern,
      name: descriptor.name,
      root: descriptor.root,
      benefit: descriptor.benefit,
      version: callback_version(opts)
    }
  end

  defp entry_declaration({:add_pattern_population, _callback, opts}) do
    %{kind: :add_pattern_population, version: callback_version(opts)}
  end

  defp callback_version(opts) do
    Keyword.get(opts, :version, :unversioned)
  end

  defp populate_or_destroy!(entries, context, target, converter, patterns, plan_timeout) do
    populate(entries, context, target, converter, patterns, plan_timeout)
  rescue
    exception ->
      MLIR.RewritePatternSet.threaded_destroy(context, patterns)
      reraise exception, __STACKTRACE__
  catch
    kind, reason ->
      MLIR.RewritePatternSet.threaded_destroy(context, patterns)
      :erlang.raise(kind, reason, __STACKTRACE__)
  end

  defp populate([], _context, _target, _converter, _patterns, _plan_timeout), do: :ok

  defp populate([entry | rest], context, target, converter, patterns, plan_timeout) do
    populate_entry(entry, context, target, converter, patterns, plan_timeout)
    populate(rest, context, target, converter, patterns, plan_timeout)
  end

  defp populate_entry({:add_legal_op, name}, _context, target, _converter, _patterns, _timeout),
    do: MLIR.ConversionTarget.add_legal_op(target, name)

  defp populate_entry({:add_illegal_op, name}, _context, target, _converter, _patterns, _timeout),
    do: MLIR.ConversionTarget.add_illegal_op(target, name)

  defp populate_entry(
         {:add_legal_dialect, name},
         _context,
         target,
         _converter,
         _patterns,
         _timeout
       ),
       do: MLIR.ConversionTarget.add_legal_dialect(target, name)

  defp populate_entry(
         {:add_illegal_dialect, name},
         _context,
         target,
         _converter,
         _patterns,
         _timeout
       ),
       do: MLIR.ConversionTarget.add_illegal_dialect(target, name)

  defp populate_entry(
         {:add_dynamically_legal_op, name, callback, _opts},
         _context,
         target,
         _converter,
         _patterns,
         _timeout
       ),
       do: MLIR.ConversionTarget.add_dynamically_legal_op(target, name, callback)

  defp populate_entry(
         {:add_dynamically_legal_dialect, name, callback, _opts},
         _context,
         target,
         _converter,
         _patterns,
         _timeout
       ),
       do: MLIR.ConversionTarget.add_dynamically_legal_dialect(target, name, callback)

  defp populate_entry(
         {:mark_recursively_legal, name, callback, _opts},
         _context,
         target,
         _converter,
         _patterns,
         _timeout
       ),
       do: MLIR.ConversionTarget.mark_recursively_legal(target, name, callback)

  defp populate_entry(
         {:mark_unknown_dynamically_legal, callback, _opts},
         _context,
         target,
         _converter,
         _patterns,
         _timeout
       ),
       do: MLIR.ConversionTarget.mark_unknown_dynamically_legal(target, callback)

  defp populate_entry(
         {:add_conversion, callback, _opts},
         _context,
         _target,
         converter,
         _patterns,
         _timeout
       ),
       do: MLIR.TypeConverter.add_conversion(converter, callback)

  defp populate_entry(
         {:add_conversion_map, source_types, target_type},
         context,
         _target,
         converter,
         _patterns,
         _timeout
       ) do
    sources = Enum.map(source_types, &MLIR.Type.get(&1, ctx: context))
    target = MLIR.Type.get(target_type, ctx: context)
    MLIR.TypeConverter.add_conversion_map(converter, sources, target)
  end

  defp populate_entry(
         {:add_1_to_n_conversion, callback, _opts},
         _context,
         _target,
         converter,
         _patterns,
         _timeout
       ),
       do: MLIR.TypeConverter.add_1_to_n_conversion(converter, callback)

  defp populate_entry(
         {:add_source_materialization, callback, _opts},
         _context,
         _target,
         converter,
         _patterns,
         _timeout
       ),
       do: MLIR.TypeConverter.add_source_materialization(converter, callback)

  defp populate_entry(
         {:add_target_materialization, callback, _opts},
         _context,
         _target,
         converter,
         _patterns,
         _timeout
       ),
       do: MLIR.TypeConverter.add_target_materialization(converter, callback)

  defp populate_entry(
         {:add_1_to_n_target_materialization, callback, _opts},
         _context,
         _target,
         converter,
         _patterns,
         _timeout
       ),
       do: MLIR.TypeConverter.add_1_to_n_target_materialization(converter, callback)

  defp populate_entry(
         {:add_conversion_pattern, root_name, callback, opts},
         context,
         _target,
         converter,
         patterns,
         plan_timeout
       ) do
    pattern_opts = conversion_pattern_opts(opts, context, plan_timeout)
    MLIR.ConversionPattern.add(patterns, root_name, converter, callback, pattern_opts)
  end

  defp populate_entry(
         {:add_pattern, descriptor, _opts},
         context,
         _target,
         _converter,
         patterns,
         _timeout
       ),
       do: MLIR.RewritePatternSet.add(patterns, descriptor, ctx: context)

  defp populate_entry(
         {:add_pattern_population, callback, _opts},
         _context,
         _target,
         converter,
         patterns,
         _timeout
       ) do
    case callback.(patterns, converter) do
      :ok -> :ok
      other -> raise ArgumentError, "pattern population must return :ok, got: #{inspect(other)}"
    end
  end

  defp conversion_pattern_opts(opts, context, plan_timeout) do
    pattern_opts =
      opts
      |> Keyword.take([:benefit, :one_to_n, :timeout])
      |> Keyword.reject(fn {key, value} -> key == :timeout and is_nil(value) end)
      |> Keyword.put(:ctx, context)

    if plan_timeout && not Keyword.has_key?(pattern_opts, :timeout) do
      Keyword.put(pattern_opts, :timeout, plan_timeout)
    else
      pattern_opts
    end
  end
end
