alias Beaver.MLIR.{ConversionTarget, RewritePatternSet, Type, TypeConverter}

defmodule Beaver.MLIR.Conversion.Plan do
  @moduledoc """
  Reusable declarations for dialect conversion: legality rules, type conversions, materializations and patterns.
  Plans hold no native handles and work across contexts. Each run creates fresh resources and destroys patterns
  before converters and targets.

  `declaration/1` omits callback bodies and runtime state. Supply callback `:version` metadata for reproducible
  declarations; absent versions are `:unversioned`.
  """

  alias Beaver.{MLIR, Native}
  alias Beaver.Pattern.Native.{Descriptor, Program}
  import Bitwise

  defstruct mode: :full, timeout: 30_000, folding_mode: nil, build_materializations: nil, entries: []

  @type mode() :: :full | :partial | :analysis
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
  Creates a plan with `:mode` (`:full`, `:partial` or `:analysis`), `:timeout` (default 30_000), `:folding_mode`
  (`:never`, `:before_patterns`, `:after_patterns` or nil) and `:build_materializations` (boolean or nil).
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

  for {name, plural, field} <- [
        {:add_legal_op, :add_legal_ops, :op},
        {:add_illegal_op, :add_illegal_ops, :op},
        {:add_legal_dialect, :add_legal_dialects, :dialect},
        {:add_illegal_dialect, :add_illegal_dialects, :dialect}
      ] do
    @spec unquote(name)(t(), String.Chars.t()) :: t()
    def unquote(name)(%__MODULE__{} = plan, name), do: append_entry(plan, {unquote(name), to_string(name)})

    @doc "Adds multiple entries via `#{name}/2`."
    @spec unquote(plural)(t(), Enumerable.t()) :: t()
    def unquote(plural)(%__MODULE__{} = plan, names), do: Enum.reduce(names, plan, &unquote(name)(&2, &1))

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

  def mark_recursively_legal(%__MODULE__{} = plan, name, opts, []) when is_list(opts),
    do: mark_recursively_legal(plan, name, nil, opts)

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
        {:add_1_to_n_target_materialization, 5, quote(do: function())},
        {:add_pattern_population, 2, quote(do: (RewritePatternSet.t(), TypeConverter.t() -> :ok))}
      ] do
    if name == :add_pattern_population do
      @doc """
      Runs once per materialization with the mutable pattern set and converter; must return `:ok`. Installed native
      patterns avoid per-match BEAM callbacks.
      """
    end

    @spec unquote(name)(t(), unquote(callback_type), keyword()) :: t()
    def unquote(name)(%__MODULE__{} = plan, callback, opts \\ []) when is_function(callback, unquote(arity)) do
      opts = validate_callback_opts!(opts, [:version], unquote(Atom.to_string(name)))
      append_entry(plan, {unquote(name), callback, opts})
    end
  end

  @doc """
  Adds a native type mapping. Assembly strings resolve in each run's context;
  the mapping has no callbacks and needs no version metadata.

  Options:
    * `:recursive` - when `true`, recursively rewrites nested occurrences of `source_types`
      inside composite types (`FunctionType`, `LLVM.struct`, `LLVM.array`, `memref`, `tensor`, `vector`).
    * `:fallback` - `:identity` (default) or `:declined`.
  """
  @spec add_conversion_map(t(), [String.t()], String.t(), keyword()) :: t()
  def add_conversion_map(%__MODULE__{} = plan, source_types, target_type, opts \\ [])
      when is_list(source_types) and is_binary(target_type) and is_list(opts) do
    unless source_types != [] and Enum.all?(source_types, &is_binary/1),
      do: raise(ArgumentError, "conversion map source types must be a non-empty list of strings")

    opts = validate_callback_opts!(opts, [:recursive, :fallback], "add_conversion_map")
    fallback = Keyword.get(opts, :fallback, :identity)
    recursive = Keyword.get(opts, :recursive, false)

    unless fallback in [:identity, :declined] and is_boolean(recursive),
      do:
        raise(
          ArgumentError,
          "add_conversion_map :fallback must be :identity or :declined and :recursive must be boolean"
        )

    if opts == [] do
      append_entry(plan, {:add_conversion_map, source_types, target_type})
    else
      append_entry(plan, {:add_conversion_map, source_types, target_type, opts})
    end
  end

  @doc """
  Populates upstream MLIR structural type conversion patterns and legality rules natively in C++ without BEAM
  callbacks.

  Supported domains are `:func` (function signatures, `func.call`, `func.return`, `llvm.func`, `llvm.return`), `:cf`
  (branch operations and `cf` dialect), and `:scf` (`scf.if`, `scf.for`, `scf.while`, `scf.yield`).
  """
  @spec add_structural_type_conversions(t(), [atom()]) :: t()
  def add_structural_type_conversions(%__MODULE__{} = plan, domains \\ [:func, :cf, :scf])
      when is_list(domains) do
    allowed_domains = [:func, :cf, :scf]

    unless domains != [] and Enum.all?(domains, &(&1 in allowed_domains)),
      do:
        raise(ArgumentError, "structural conversion domains must be a non-empty subset of #{inspect(allowed_domains)}")

    append_entry(plan, {:add_structural_type_conversions, Enum.uniq(domains)})
  end

  @type conversion_callback() ::
          MLIR.ConversionPattern.callback()
          | (MLIR.Operation.t(),
             MLIR.ConversionPattern.adaptor(),
             MLIR.ConversionPatternRewriter.t(),
             TypeConverter.t() ->
               term())

  @spec add_conversion_pattern(t(), String.Chars.t() | [String.Chars.t()], conversion_callback(), keyword()) :: t()
  def add_conversion_pattern(plan, root_or_roots, callback, opts \\ [])

  def add_conversion_pattern(%__MODULE__{} = plan, [_ | _] = roots, callback, opts)
      when is_function(callback, 3) or is_function(callback, 4) do
    Enum.reduce(roots, plan, &add_conversion_pattern(&2, &1, callback, opts))
  end

  def add_conversion_pattern(%__MODULE__{} = plan, root_name, callback, opts)
      when (is_binary(root_name) or is_atom(root_name)) and
             (is_function(callback, 3) or is_function(callback, 4)) do
    opts = validate_conversion_pattern_opts!(opts)
    append_entry(plan, {:add_conversion_pattern, to_string(root_name), callback, opts})
  end

  defp validate_conversion_pattern_opts!(opts) do
    opts = validate_callback_opts!(opts, [:version, :benefit, :one_to_n, :timeout], "add_conversion_pattern")

    validate_non_neg_int_opt!(Keyword.get(opts, :benefit, 1), :benefit)
    validate_boolean_opt!(Keyword.get(opts, :one_to_n, false), :one_to_n)
    validate_optional_timeout!(Keyword.get(opts, :timeout))
    opts
  end

  defp validate_non_neg_int_opt!(v, _key) when is_integer(v) and v >= 0, do: :ok

  defp validate_non_neg_int_opt!(v, key),
    do: raise(ArgumentError, "#{inspect(key)} must be a non-negative integer, got: #{inspect(v)}")

  defp validate_boolean_opt!(v, _key) when is_boolean(v), do: :ok
  defp validate_boolean_opt!(v, key), do: raise(ArgumentError, "#{inspect(key)} must be boolean, got: #{inspect(v)}")

  defp validate_optional_timeout!(nil), do: :ok
  defp validate_optional_timeout!(v) when is_integer(v) and v >= 0, do: :ok

  defp validate_optional_timeout!(v),
    do: raise(ArgumentError, ":timeout must be a non-negative integer or nil, got: #{inspect(v)}")

  @spec add_pattern(t(), Descriptor.t() | Program.t(), keyword()) :: t()
  def add_pattern(plan, pattern, opts \\ [])

  def add_pattern(%__MODULE__{} = plan, %Descriptor{} = descriptor, opts) do
    opts = validate_callback_opts!(opts, [:version], "add_pattern")
    append_entry(plan, {:add_pattern, descriptor, opts})
  end

  def add_pattern(%__MODULE__{} = plan, %Program{} = program, opts) do
    opts = validate_callback_opts!(opts, [:version], "add_pattern")
    append_entry(plan, {:add_program, program, opts})
  end

  @doc "Adds a callback-free C ABI aggregate call lowering pattern to the plan."
  @spec add_c_abi_call_pattern(t(), String.t(), String.t(), keyword()) :: t()
  def add_c_abi_call_pattern(%__MODULE__{} = plan, root_op, target_op \\ "llvm.call", opts \\ [])
      when is_binary(root_op) and is_binary(target_op) and is_list(opts) do
    {plan_opts, call_opts} = Keyword.split(opts, [:version])
    program = Program.c_abi_call(root_op, target_op, call_opts)
    add_pattern(plan, program, plan_opts)
  end

  @doc """
  Adds PDL patterns (as a PDL MLIR text string or list of builder callbacks) to the plan. When materialized, the PDL
  patterns are compiled to `PDLInterp` bytecode and executed natively without BEAM callbacks.
  """
  @spec add_pdl_patterns(t(), binary() | [function()], keyword()) :: t()
  def add_pdl_patterns(%__MODULE__{} = plan, pdl_source_or_builders, opts \\ [])
      when (is_binary(pdl_source_or_builders) or is_list(pdl_source_or_builders)) and is_list(opts) do
    opts = validate_callback_opts!(opts, [:version], "add_pdl_patterns")
    append_entry(plan, {:add_pdl_patterns, pdl_source_or_builders, opts})
  end

  @doc """
  Returns configuration and declaration metadata, omitting callback bodies and runtime state. Reproducibility
  requires explicit callback `:version` metadata.
  """
  @spec declaration(t()) :: map()
  def declaration(%__MODULE__{} = plan),
    do: plan |> Map.from_struct() |> Map.update!(:entries, &Enum.map(&1, fn entry -> entry_declaration(entry) end))

  @doc "Executes on a module or operation with fresh, scoped native resources. Returns `MLIR.Conversion.result()`."
  @spec run(t(), MLIR.Conversion.conversion_ir()) :: MLIR.Conversion.result()
  def run(%__MODULE__{} = plan, ir), do: execute(plan, ir, false)

  @doc "Returns `{run_result, receipt}` with bounded conversion callback profiling."
  @spec profile(t(), MLIR.Conversion.conversion_ir()) :: {MLIR.Conversion.result(), MLIR.Conversion.Profile.receipt()}
  def profile(%__MODULE__{} = plan, ir), do: execute(plan, ir, true)

  @doc "Runs with explicit mutation and legalization reporting."
  def report(%__MODULE__{} = plan, ir), do: execute(plan, ir, :report)

  @doc "Runs legalization analysis without committing IR changes."
  def analyze(%__MODULE__{} = plan, ir), do: report(%{plan | mode: :analysis}, ir)

  defp execute(plan, ir, profile?) do
    context = MLIR.context(ir)
    target_opts = if plan.timeout, do: [timeout: plan.timeout], else: []

    opts = Map.take(plan, [:timeout, :folding_mode, :build_materializations]) |> Enum.reject(fn {_, v} -> is_nil(v) end)

    function =
      case profile? do
        true -> :profile
        :report -> :run
        false -> :apply
      end

    Native.with_resource(ConversionTarget.create(context, target_opts), &ConversionTarget.destroy/1, fn target ->
      TypeConverter.with(target_opts, fn converter ->
        patterns = RewritePatternSet.create(context)
        populate_or_destroy!(plan.entries, context, target, converter, patterns, plan.timeout)
        # Conversion owns the frozen set once submitted, including worker failure.
        apply(MLIR.Conversion, function, [plan.mode, ir, target, patterns, opts])
      end)
    end)
  end

  @doc "Executes the conversion plan on the given IR, returning the converted IR or raising `MLIR.Conversion.Error`."
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

  @doc """
  Executes `plan` on `ir`, optionally runs `"reconcile-unrealized-casts"` (`reconcile_casts: true` by default), and
  verifies that no `"builtin.unrealized_conversion_cast"` operations remain (unless `allow_unrealized_casts: true`).
  """
  @spec finish(t(), MLIR.Conversion.conversion_ir(), keyword()) ::
          {:ok, MLIR.Conversion.conversion_ir(), [term()]} | {:error, term()}
  def finish(%__MODULE__{} = plan, ir, opts \\ []) when is_list(opts) do
    opts = validate_callback_opts!(opts, [:reconcile_casts, :allow_unrealized_casts], "Plan.finish")
    reconcile? = Keyword.get(opts, :reconcile_casts, true)
    allow_casts? = Keyword.get(opts, :allow_unrealized_casts, false)

    with {:ok, converted, diagnostics} <- run(plan, ir),
         {:ok, reconciled} <- maybe_reconcile_casts(converted, reconcile?),
         :ok <- check_unrealized_casts(reconciled, allow_casts?) do
      {:ok, reconciled, diagnostics}
    end
  end

  @doc "Executes `finish/3` and returns the finalized IR or raises on failure."
  @spec finish!(t(), MLIR.Conversion.conversion_ir(), keyword()) :: MLIR.Conversion.conversion_ir()
  def finish!(%__MODULE__{} = plan, ir, opts \\ []) do
    case finish(plan, ir, opts) do
      {:ok, converted, _diagnostics} -> converted
      {:error, %MLIR.Conversion.Error{} = error} -> raise error
      {:error, reason} when is_binary(reason) -> raise ArgumentError, reason
      {:error, reason} -> raise ArgumentError, "Plan.finish! failed: #{inspect(reason)}"
    end
  end

  defp maybe_reconcile_casts(ir, false), do: {:ok, ir}

  defp maybe_reconcile_casts(ir, true) do
    case ir |> Beaver.Composer.append("reconcile-unrealized-casts") |> Beaver.Composer.run() do
      {:ok, reconciled} -> {:ok, reconciled}
      {:ok, reconciled, _diagnostics} -> {:ok, reconciled}
      {:error, diagnostics} -> {:error, MLIR.Diagnostic.format(diagnostics, "reconcile-unrealized-casts failed")}
    end
  end

  defp check_unrealized_casts(_ir, true), do: :ok

  defp check_unrealized_casts(ir, false) do
    case Beaver.Walker.find_ops(ir, "builtin.unrealized_conversion_cast") do
      [] ->
        :ok

      casts ->
        {:error, "#{length(casts)} residual builtin.unrealized_conversion_cast operation(s) remained after conversion"}
    end
  end

  defp append_entry(%__MODULE__{entries: entries} = plan, entry), do: %{plan | entries: entries ++ [entry]}

  defp validate_plan_options!(opts),
    do: validate_callback_opts!(opts, [:mode, :timeout, :folding_mode, :build_materializations], "Plan")

  defp validate_mode!(mode) when mode in [:full, :partial, :analysis], do: mode

  defp validate_mode!(mode),
    do: raise(ArgumentError, ":mode must be :full, :partial or :analysis, got: #{inspect(mode)}")

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

  defp entry_declaration({kind, _callback, opts}) when kind in ~w(
         mark_unknown_dynamically_legal add_conversion add_1_to_n_conversion add_source_materialization
         add_target_materialization add_1_to_n_target_materialization add_pattern_population
       )a,
    do: %{kind: kind, version: callback_version(opts)}

  defp entry_declaration({:add_conversion_map, source_types, target_type}),
    do: %{kind: :add_conversion_map, source_types: source_types, target_type: target_type}

  defp entry_declaration({:add_conversion_map, source_types, target_type, opts}),
    do: Map.merge(%{kind: :add_conversion_map, source_types: source_types, target_type: target_type}, Map.new(opts))

  defp entry_declaration({:add_structural_type_conversions, domains}),
    do: %{kind: :add_structural_type_conversions, domains: domains}

  defp entry_declaration({:add_conversion_pattern, root, _cb, opts}),
    do: %{
      kind: :add_conversion_pattern,
      root: root,
      benefit: Keyword.get(opts, :benefit, 1),
      one_to_n: Keyword.get(opts, :one_to_n, false),
      timeout: Keyword.get(opts, :timeout),
      version: callback_version(opts)
    }

  defp entry_declaration({kind, pattern, opts}) when kind in [:add_pattern, :add_program] do
    fields = if kind == :add_pattern, do: [:name, :root, :benefit], else: [:root, :benefit]
    Map.merge(Map.take(pattern, fields), %{kind: kind, version: callback_version(opts)})
  end

  defp entry_declaration({:add_pdl_patterns, _source, opts}),
    do: %{kind: :add_pdl_patterns, version: callback_version(opts)}

  defp callback_version(opts), do: Keyword.get(opts, :version, :unversioned)

  defp populate_or_destroy!(entries, ctx, target, converter, patterns, timeout) do
    has_type_conversions =
      Enum.any?(entries, fn
        {:add_conversion, _, _} -> true
        {:add_1_to_n_conversion, _, _} -> true
        {:add_conversion_map, _, _} -> true
        {:add_conversion_map, _, _, _} -> true
        _ -> false
      end)

    if not has_type_conversions do
      i1 = Type.i1(ctx: ctx)
      TypeConverter.add_conversion_map(converter, [i1], i1, fallback: :identity)
    end

    resources = %{
      context: ctx,
      target: target,
      converter: converter,
      program_converter: if(has_type_conversions, do: converter, else: nil),
      patterns: patterns,
      timeout: timeout
    }

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

  defp populate_entry({kind, callback, _opts}, %{converter: converter}) when kind in ~w(
         add_conversion add_1_to_n_conversion add_source_materialization add_target_materialization
         add_1_to_n_target_materialization
       )a,
    do: apply(TypeConverter, kind, [converter, callback])

  defp populate_entry({:add_conversion_map, sources, target}, resources),
    do: populate_entry({:add_conversion_map, sources, target, []}, resources)

  defp populate_entry({:add_conversion_map, source_types, target_type, opts}, resources) do
    sources = Enum.map(source_types, &Type.get(&1, ctx: resources.context))
    target = Type.get(target_type, ctx: resources.context)
    TypeConverter.add_conversion_map(resources.converter, sources, target, opts)
  end

  defp populate_entry({:add_structural_type_conversions, domains}, resources) do
    domain_bits = %{func: 1, cf: 2, scf: 4}
    mask = Enum.reduce(domains, 0, fn d, acc -> acc ||| Map.fetch!(domain_bits, d) end)

    :ok =
      MLIR.CAPI.beaver_raw_conversion_populate_structural(
        resources.target.registration,
        resources.patterns.ref,
        resources.converter.registration,
        mask
      )
  end

  defp populate_entry({:add_conversion_pattern, root_name, callback, opts}, resources) do
    opts = conversion_pattern_opts(opts, resources.context, resources.timeout)

    wrapped =
      if is_function(callback, 4),
        do: fn op, adaptor, rw -> callback.(op, adaptor, rw, resources.converter) end,
        else: callback

    MLIR.ConversionPattern.add(resources.patterns, root_name, resources.converter, wrapped, opts)
  end

  defp populate_entry({:add_pattern, descriptor, _opts}, resources),
    do: RewritePatternSet.add(resources.patterns, descriptor, ctx: resources.context)

  defp populate_entry({:add_program, program, _opts}, resources),
    do:
      RewritePatternSet.add(resources.patterns, program,
        ctx: resources.context,
        type_converter: resources.program_converter
      )

  defp populate_entry({:add_pdl_patterns, pdl_source_or_builders, _opts}, resources),
    do: RewritePatternSet.add_pdl(resources.patterns, pdl_source_or_builders, ctx: resources.context)

  defp populate_entry({:add_pattern_population, callback, _opts}, resources) do
    case callback.(resources.patterns, resources.converter) do
      :ok -> :ok
      other -> raise ArgumentError, "pattern population must return :ok, got: #{inspect(other)}"
    end
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
