defmodule Beaver.MLIR.CompilationRuntime do
  @moduledoc """
  Incremental MLIR compilation for long-running BEAM applications.

  The runtime caches transformed, versioned MLIR bytecode. Cache lookup uses a
  fast source digest plus every compatibility input; the stored artifact key
  additionally contains MLIR's structural hash. A hit therefore skips parsing
  and the transform pipeline, while JIT and object emission consume the same
  validated bytecode.

  ## Cache compatibility

  The following inputs invalidate an artifact automatically:

    * source content and structural hash;
    * the LLVM revision used to build Beaver;
    * pass/transform pipeline identity;
    * target configuration;
    * dynamic dialect/schema version;
    * requested bytecode emit version.

  Custom pass functions are not stable cache identities. Supply an explicit
  `:pipeline_version` whenever `:pipeline` contains runtime functions.

  ## Telemetry

  If the `:telemetry` library is loaded, events are emitted below
  `[:beaver, :mlir, :compilation, ...]`. A three-argument callback can also be
  supplied in the `:telemetry` option. Durations use native time units.
  """

  alias Beaver.Composer
  alias Beaver.MLIR
  alias MLIR.CompilationCache
  alias __MODULE__.{Artifact, CacheKey}

  @entry_format 1

  @type compile_option ::
          {:cache, CompilationCache.cache()}
          | {:pipeline, term()}
          | {:pipeline_version, term()}
          | {:target, term()}
          | {:schema_version, term()}
          | {:desired_emit_version, integer() | nil}
          | {:context, MLIR.Context.t()}
          | {:context_options, keyword()}
          | {:llvm_revision, String.t()}
          | {:telemetry, (list(atom()), map(), map() -> any())}

  @spec llvm_revision() :: String.t()
  def llvm_revision do
    MLIR.CAPI.beaverGetLLVMVersion() |> MLIR.to_string()
  end

  @spec compile(binary() | MLIR.Module.t(), [compile_option()]) ::
          {:ok, Artifact.t()} | {:error, Exception.t()}
  def compile(source, opts \\ []) do
    {:ok, compile!(source, opts)}
  rescue
    exception -> {:error, exception}
  end

  @spec compile!(binary() | MLIR.Module.t(), [compile_option()]) :: Artifact.t()
  def compile!(source, opts \\ []) do
    cache = Keyword.get(opts, :cache, :memory)
    {source_bytes, source_context} = source_bytes_and_context(source)
    inputs = compatibility_inputs(source_bytes, opts)
    lookup_key = CacheKey.lookup(inputs)

    {cached, cache_lookup_duration} = timed(fn -> CompilationCache.get(cache, lookup_key) end)

    MLIR.Telemetry.emit(
      [:cache, :lookup],
      %{duration: cache_lookup_duration},
      %{lookup_key: lookup_key},
      opts
    )

    case validate_cache_entry(cached, lookup_key, inputs) do
      {:ok, entry} ->
        timings = %{cache_lookup: cache_lookup_duration, parse: 0, transform: 0, serialize: 0}
        emit_cache_result(:hit, lookup_key, timings, opts)
        artifact_from_entry(entry, :hit, timings)

      {:miss, reason} ->
        if reason != :not_found do
          CompilationCache.delete(cache, lookup_key)

          MLIR.Telemetry.emit(
            [:cache, :failure],
            %{count: 1},
            %{lookup_key: lookup_key, reason: reason},
            opts
          )
        end

        compile_miss(
          source_bytes,
          source_context,
          inputs,
          lookup_key,
          cache,
          cache_lookup_duration,
          opts
        )
    end
  end

  @doc "Explicitly invalidate one cache lookup key, or all entries."
  @spec invalidate(CompilationCache.cache(), String.t() | :all) :: :ok | {:error, term()}
  def invalidate(cache \\ :memory, key \\ :all)
  def invalidate(cache, :all), do: CompilationCache.clear(cache)
  def invalidate(cache, key) when is_binary(key), do: CompilationCache.delete(cache, key)

  @doc "Create and initialize a JIT from a validated compilation artifact."
  @spec jit!(Artifact.t(), keyword()) :: MLIR.ExecutionEngine.t()
  def jit!(%Artifact{} = artifact, opts \\ []) do
    {jit, codegen_duration} =
      timed(fn ->
        with_module(artifact, opts, fn module ->
          MLIR.ExecutionEngine.create!(module, execution_engine_opts(opts))
        end)
      end)

    MLIR.Telemetry.emit(
      [:codegen],
      %{duration: codegen_duration},
      event_metadata(artifact),
      opts
    )

    if Keyword.get(opts, :initialize, true) do
      {jit, load_duration} = timed(fn -> MLIR.ExecutionEngine.init(jit) end)

      MLIR.Telemetry.emit(
        [:jit_load],
        %{duration: load_duration},
        event_metadata(artifact),
        opts
      )

      jit
    else
      jit
    end
  end

  @doc "Emit an object file from the same normalized artifact used by `jit!/2`."
  @spec emit_object!(Artifact.t(), Path.t(), keyword()) :: Path.t()
  def emit_object!(%Artifact{} = artifact, path, opts \\ []) do
    opts = Keyword.merge([object_dump: true, enable_pic: true], opts)
    jit = jit!(artifact, Keyword.put(opts, :initialize, false))

    try do
      {path, duration} = timed(fn -> MLIR.ExecutionEngine.emit_object!(jit, path) end)

      MLIR.Telemetry.emit(
        [:object_emission],
        %{duration: duration},
        Map.put(event_metadata(artifact), :path, path),
        opts
      )

      path
    after
      MLIR.ExecutionEngine.destroy(jit)
    end
  end

  defp compile_miss(
         source_bytes,
         source_context,
         inputs,
         lookup_key,
         cache,
         cache_lookup_duration,
         opts
       ) do
    {context, owned_context?} = compilation_context(source_context, opts)

    try do
      {module, parse_duration} =
        timed(fn -> MLIR.Bytecode.read!(source_bytes, ctx: context) end)

      try do
        structural_hash =
          module
          |> MLIR.Operation.from_module()
          |> MLIR.Operation.structural_hash(ignore_locations: true)

        {module, transform_duration} = timed(fn -> run_pipeline(module, opts) end)

        {bytecode, serialize_duration} =
          timed(fn ->
            MLIR.Bytecode.write!(module,
              desired_emit_version: Keyword.get(opts, :desired_emit_version)
            )
          end)

        artifact_key = CacheKey.artifact(inputs, structural_hash)

        metadata =
          Map.merge(inputs, %{
            structural_hash: structural_hash,
            lookup_key: lookup_key,
            artifact_key: artifact_key
          })

        entry = %{
          format: @entry_format,
          lookup_key: lookup_key,
          artifact_key: artifact_key,
          metadata: metadata,
          bytecode: bytecode,
          bytecode_digest: digest(bytecode)
        }

        case CompilationCache.put(cache, lookup_key, entry) do
          :ok ->
            :ok

          {:error, reason} ->
            MLIR.Telemetry.emit(
              [:cache, :failure],
              %{count: 1},
              %{lookup_key: lookup_key, reason: {:write, reason}},
              opts
            )
        end

        timings = %{
          cache_lookup: cache_lookup_duration,
          parse: parse_duration,
          transform: transform_duration,
          serialize: serialize_duration
        }

        emit_stage(:parse, parse_duration, artifact_key, opts)
        emit_stage(:transform, transform_duration, artifact_key, opts)
        emit_stage(:serialize, serialize_duration, artifact_key, opts)
        emit_cache_result(:miss, lookup_key, timings, opts)

        artifact_from_entry(entry, :miss, timings)
      after
        MLIR.Module.destroy(module)
      end
    after
      if owned_context?, do: MLIR.Context.destroy(context)
    end
  end

  defp source_bytes_and_context(%MLIR.Module{} = module) do
    {MLIR.Bytecode.write!(module), MLIR.context(module)}
  end

  defp source_bytes_and_context(source) when is_binary(source), do: {source, nil}

  defp compilation_context(source_context, opts) do
    case Keyword.get(opts, :context, source_context) do
      nil -> {MLIR.Context.create(Keyword.get(opts, :context_options, [])), true}
      context -> {context, false}
    end
  end

  defp compatibility_inputs(source_bytes, opts) do
    %{
      source_digest: digest(source_bytes),
      llvm_revision: Keyword.get_lazy(opts, :llvm_revision, &llvm_revision/0),
      pipeline: pipeline_identity(opts),
      target: Keyword.get(opts, :target, %{}),
      schema_version: Keyword.get(opts, :schema_version, :static),
      bytecode_version: Keyword.get(opts, :desired_emit_version, :current)
    }
  end

  defp pipeline_identity(opts) do
    case Keyword.fetch(opts, :pipeline_version) do
      {:ok, version} -> version
      :error -> assert_deterministic_pipeline!(Keyword.get(opts, :pipeline, []))
    end
  end

  defp assert_deterministic_pipeline!(pipeline) when is_function(pipeline) do
    raise ArgumentError, ":pipeline_version is required for function-based pipelines"
  end

  defp assert_deterministic_pipeline!(pipeline) when is_list(pipeline) do
    Enum.map(pipeline, &assert_deterministic_pipeline!/1)
  end

  defp assert_deterministic_pipeline!({operation, passes}) do
    {operation, assert_deterministic_pipeline!(passes)}
  end

  defp assert_deterministic_pipeline!(pipeline)
       when is_binary(pipeline) or is_atom(pipeline),
       do: pipeline

  defp assert_deterministic_pipeline!(pipeline) do
    raise ArgumentError,
          ":pipeline_version is required for non-data pipeline #{inspect(pipeline, limit: 5)}"
  end

  defp run_pipeline(module, opts) do
    case Keyword.get(opts, :pipeline, []) do
      [] ->
        module

      pipeline ->
        pipeline
        |> List.wrap()
        |> Enum.reduce(Composer.new(module), &Composer.append(&2, &1))
        |> Composer.run!()
    end
  end

  defp validate_cache_entry(:miss, _lookup_key, _inputs), do: {:miss, :not_found}
  defp validate_cache_entry({:error, reason}, _lookup_key, _inputs), do: {:miss, {:read, reason}}

  defp validate_cache_entry({:ok, entry}, lookup_key, inputs) do
    with true <- entry[:format] == @entry_format,
         true <- entry[:lookup_key] == lookup_key,
         %{structural_hash: structural_hash} = metadata <- entry[:metadata],
         true <- Map.take(metadata, Map.keys(inputs)) == inputs,
         true <- entry[:artifact_key] == CacheKey.artifact(inputs, structural_hash),
         bytecode when is_binary(bytecode) <- entry[:bytecode],
         true <- entry[:bytecode_digest] == digest(bytecode) do
      {:ok, entry}
    else
      _ -> {:miss, :incompatible_or_corrupt_entry}
    end
  rescue
    _ -> {:miss, :incompatible_or_corrupt_entry}
  end

  defp artifact_from_entry(entry, cache_status, timings) do
    %Artifact{
      key: entry.artifact_key,
      bytecode: entry.bytecode,
      metadata: entry.metadata,
      cache: cache_status,
      timings: timings
    }
  end

  defp with_module(artifact, opts, fun) do
    context = MLIR.Context.create(Keyword.get(opts, :context_options, []))

    try do
      MLIR.Context.register_translations(context)
      module = MLIR.Bytecode.read!(artifact.bytecode, ctx: context)

      try do
        fun.(module)
      after
        MLIR.Module.destroy(module)
      end
    after
      MLIR.Context.destroy(context)
    end
  end

  defp execution_engine_opts(opts) do
    Keyword.take(opts, [:shared_lib_paths, :opt_level, :object_dump, :enable_pic, :dirty])
  end

  defp emit_stage(stage, duration, artifact_key, opts) do
    MLIR.Telemetry.emit([stage], %{duration: duration}, %{artifact_key: artifact_key}, opts)
  end

  defp emit_cache_result(result, lookup_key, timings, opts) do
    MLIR.Telemetry.emit(
      [:cache, result],
      %{count: 1},
      %{lookup_key: lookup_key, timings: timings},
      opts
    )
  end

  defp event_metadata(artifact), do: %{artifact_key: artifact.key, cache: artifact.cache}

  defp digest(binary), do: :crypto.hash(:sha256, binary) |> Base.encode16(case: :lower)

  defp timed(fun) do
    started = System.monotonic_time()
    result = fun.()
    {result, System.monotonic_time() - started}
  end
end
