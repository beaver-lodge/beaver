defmodule Beaver.Build.Compiler do
  @moduledoc """
  LLVM compiler facade over caller-supplied templates, lowering rules, graph builders and image policy.
  Inputs expose `functions` and per-function `identity`; `leaf_dependencies/2` supplies the language-specific call graph.
  Unit builders receive owned contexts. Cache keys track the protocol, generators, target and profile.
  """
  alias Beaver.{Composer, MLIR}
  alias Beaver.Build.ArtifactCache
  alias Beaver.MLIR.{ExecutionEngine, Target.LLVMIR, Template}
  @object_options ~w(triple cpu features opt_level pic format)a
  @aot_options [:functions, :cache_dir, :max_concurrency | @object_options]
  defdelegate sha256(data), to: ArtifactCache, as: :digest
  defdelegate platform(), to: LLVMIR, as: :host_platform

  defmacro __using__(opts) do
    quote do
      @compiler_config Map.new(unquote(opts))
      @compiler_config Map.update!(@compiler_config, :generators, &[unquote(__MODULE__) | &1])
      @compiler_image @compiler_config.image
      defdelegate sha256(data), to: unquote(__MODULE__)
      defdelegate platform(), to: unquote(__MODULE__)
      defdelegate target_descriptor(), to: unquote(__MODULE__)
      defdelegate build_image!(source, output, opts \\ []), to: @compiler_image
      defdelegate image_plan(source, output, opts \\ []), to: @compiler_image, as: :plan
      defdelegate link_shared!(objects, directory, symbol, opts \\ []), to: unquote(__MODULE__)

      def compiler_identity(profile \\ @compiler_config.default_profile),
        do: unquote(__MODULE__).compiler_identity(@compiler_config, profile)

      def compile_ssa_program!(ssa, opts \\ []) do
        config = Map.put(@compiler_config, :leaf_dependencies, &leaf_dependencies/2)
        unquote(__MODULE__).compile_ssa_program!(config, ssa, opts)
      end

      unquote_splicing(
        for {name, arity} <- [
              compile_functions!: 1,
              compile_object!: 1,
              lower_to_llvm!: 1,
              jit!: 1,
              compile_mlir_unit!: 1,
              cached_object!: 3
            ] do
          args = Macro.generate_arguments(arity, __MODULE__)

          quote do
            def unquote(name)(unquote_splicing(args), opts \\ []),
              do: unquote(__MODULE__).unquote(name)(@compiler_config, unquote_splicing(args), opts)
          end
        end
      )
    end
  end

  def target_descriptor do
    target = LLVMIR.host_target()
    "#{target.triple}\n#{target.cpu}\n#{target.features}\n"
  end

  def compiler_identity(config, profile) do
    sha256([
      config.protocol.digest(),
      config.patterns.declaration().pdl_sha256,
      Enum.map(config.generators, & &1.__info__(:md5)),
      LLVMIR.target_descriptor(opt_level: level(config, profile: profile)),
      "#{profile}#{config.identity}"
    ])
  end

  def compile_functions!(config, module, opts \\ []),
    do: LLVMIR.compile_functions!(as_llvm(config, module, opts), Keyword.take(opts, @aot_options))

  def compile_object!(config, module, opts \\ []),
    do: LLVMIR.compile_object!(as_llvm(config, module, opts), Keyword.take(opts, @object_options))

  def lower_to_llvm!(config, module, opts \\ []) do
    for bundle <- opts[:bundles] || [] do
      {symbol, options} = if is_tuple(bundle), do: bundle, else: {bundle, []}
      config.builder.bundle!(module, to_string(symbol), options)
    end

    module =
      if opts[config.skip_lower_option],
        do: module,
        else: config.patterns.lower!(module, Keyword.take(opts, config.lower_options))

    config |> run_passes!(module) |> MLIR.verify!()
  end

  def jit!(config, module, opts \\ []) do
    engine =
      ExecutionEngine.create!(
        as_llvm(config, module, opts),
        Keyword.take(opts, [:shared_lib_paths, :opt_level, :symbols])
      )

    if opts[:library] && opts[:runtime_namespace],
      do: config.runtime.bind_library!(engine, opts[:library], opts[:runtime_namespace]),
      else: engine
  end

  def compile_mlir_unit!(config, source, opts \\ []) do
    start = now()

    [unit] =
      LLVMIR.compile_units!(
        [{"unit", "unit", config.templates.with_declarations(source)}],
        unit_options(config, nil, opts)
      )

    metrics = Map.take(unit.metrics, ~w(translate_ns opt_ns codegen_ns)a)
    {unit.object, Map.put(metrics, :lower_ns, now() - start)}
  end

  def cached_object!(config, cache, key, source, opts \\ []) do
    start = now()
    [unit] = LLVMIR.compile_units!([{key, key, source}], unit_options(config, cache, opts))
    %{path: unit.path, report: unit_report(unit, elapsed(start))}
  end

  def link_shared!(objects, directory, symbol, opts \\ []) do
    LLVMIR.link_shared!(
      objects,
      Path.join(directory, symbol <> LLVMIR.shared_library_extension()),
      Keyword.take(opts, [:cc, :ldflags, :libs, :undefined_lookup])
    )
  end

  def compile_ssa_program!(config, ssa, opts \\ []) when is_map(ssa) do
    start = now()
    functions = Map.fetch!(ssa, "functions")
    profile = opts[:profile] || config.default_profile
    program_digest = sha256(:erlang.term_to_binary(ssa))
    compiler = opts[:compiler_id] || compiler_identity(config, profile)
    cache = opts[:cache_dir] || Path.expand("tmp/native-cache")
    key = &sha256([compiler, :erlang.term_to_binary(&1)])
    symbol = Template.check_ident!(opts[:symbol] || "#{config.symbol_prefix}#{program_digest}")
    base_keys = Enum.map(functions, key)

    leaves =
      Enum.zip_with(functions, base_keys, &if(config.analysis.bounded_leaf?(&1), do: "#{config.unit_prefix}#{&2}_leaf"))

    leaf_table = List.to_tuple(leaves)
    dependencies = Enum.map(functions, &config.leaf_dependencies.(&1, leaf_table))

    dependent_key = fn [function, dependencies, base] ->
      if dependencies == [], do: base, else: key.({function, dependencies})
    end

    keys = Enum.zip_with([functions, dependencies, base_keys], dependent_key)
    names = Enum.map(keys, &"#{config.unit_prefix}#{&1}")

    units =
      Enum.zip_with([functions, keys, names, dependencies], fn [function, digest, name, dependencies] ->
        symbols = Enum.map(dependencies, &elem(&1, 1))
        {function["identity"], digest, &config.builder.build_unit_module!(&1, function, name, leaves, symbols)}
      end)

    dispatch = {"dispatch", key.({symbol, keys}), &config.builder.build_dispatch_module!(&1, symbol, names, leaves)}
    options = Keyword.put_new(opts, :max_concurrency, System.schedulers_online())
    compiled = LLVMIR.compile_units!(units ++ [dispatch], unit_options(config, cache, options))

    %{
      program: program_digest,
      symbol: symbol,
      compiler: compiler,
      target: target_descriptor(),
      profile: to_string(profile),
      elapsed_ms: elapsed(start),
      units: Enum.map(compiled, &[&1.id, unit_report(&1, 0.0)]),
      objects: compiled |> Enum.map(& &1.path) |> Enum.uniq()
    }
  end

  defp unit_report(unit, elapsed) do
    metrics = unit.metrics || %{}

    fields = [
      emit_ms: :emit_ns,
      lower_ms: :lower_ns,
      translate_ms: :translate_ns,
      parse_ms: :parse_ns,
      optimize_ms: :opt_ns,
      codegen_ms: :codegen_ns
    ]

    phases =
      for {name, key} <- fields,
          into: %{},
          do: {to_string(name), Map.get(metrics, key, Map.get(metrics, to_string(key), 0)) / 1.0e6}

    build = Map.put(phases, "peak_rss_bytes", 0)

    %{
      "key" => unit.key,
      "cache_hit" => unit.cached?,
      "elapsed_ms" => elapsed,
      "object_bytes" => byte_size(unit.object),
      "build" => build
    }
  end

  defp unit_options(config, cache, opts),
    do: [
      cache_dir: cache,
      lower: &lower_guest!(config, &1),
      passes: config.passes,
      opt_level: level(config, opts),
      pic: true,
      max_concurrency: opts[:max_concurrency] || 1
    ]

  defp level(config, opts), do: config.profiles[opts[:profile]] || opts[:opt_level] || 2

  defp lower_guest!(config, module),
    do: module |> config.templates.ensure_declarations!() |> config.patterns.lower!(import_runtime: true)

  defp run_passes!(config, module), do: Enum.reduce(config.passes, module, &Composer.append(&2, &1)) |> Composer.run!()

  defp as_llvm(config, module, opts),
    do: if(opts[:already_llvm], do: module, else: lower_to_llvm!(config, module, opts))

  defp now, do: System.monotonic_time(:nanosecond)
  defp elapsed(start), do: (now() - start) / 1.0e6
end
