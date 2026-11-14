defmodule Beaver.Build.Image do
  @moduledoc "Cached image plans over caller-supplied preparation, analysis, compilation, linking and manifest policies."
  alias Beaver.Build.{ArtifactCache, Plan}

  def plan(source, output, opts, policy) do
    Plan.new()
    |> Plan.stage(:read, [], fn _ -> read(source, opts, policy) end)
    |> Plan.stage(:analyze, [:read], &analyze(&1.read, policy))
    |> Plan.stage(:compile, [:read, :analyze], &compile(&1.read, &1.analyze, opts, policy))
    |> Plan.stage(:link, [:read, :compile], &link(&1.read, &1.compile, opts, policy))
    |> Plan.stage(:publish, [:read, :analyze, :compile, :link], &publish(&1, output, policy))
  end

  defp read(source, opts, policy) do
    programs = policy.prepare.(source)
    backend = opts |> Keyword.get(:backend, :llvm) |> to_string() |> String.downcase()
    unless backend in ~w(llvm interp), do: raise(ArgumentError, "unsupported image backend: #{backend}")
    cache = opts[:cache_dir] || System.get_env(policy.cache_env) || Path.expand("tmp/image-cache")
    compiler = opts[:compiler_id] || policy.compiler_id.(opts[:profile] || :balanced)
    source_digest = ArtifactCache.digest(JSON.encode!(programs))
    metadata = if is_function(policy.metadata, 1), do: policy.metadata.(programs), else: policy.metadata.()
    key = ArtifactCache.digest(JSON.encode!([metadata["version"], source_digest, compiler, backend]))
    directory = Path.join([cache, "images", key])
    bundle = ArtifactCache.bundle(directory, policy.manifest, policy.checksum)
    native_cache = opts[:native_cache_dir] || System.get_env(policy.native_cache_env) || Path.join(cache, "native")
    caches = %{analysis_cache: Path.join(cache, "analysis"), native_cache: native_cache}
    identity = %{programs: programs, backend: backend, compiler: compiler, source: source_digest, key: key}
    bundle |> Map.merge(caches) |> Map.merge(identity) |> Map.put(:metadata, metadata)
  end

  defp analyze(%{cached: cached}, _) when not is_nil(cached), do: %{programs: cached["programs"], count: 0}

  defp analyze(input, policy) do
    {programs, count} =
      ArtifactCache.map_json!(input.programs, input.analysis_cache, input.compiler, policy.ready?, policy.analyze)

    %{programs: programs, count: count}
  end

  defp compile(%{cached: cached}, _, _, _) when not is_nil(cached), do: []
  defp compile(%{backend: "interp"}, _, _, _), do: []

  defp compile(input, analysis, opts, policy) do
    options = Keyword.merge(opts, cache_dir: input.native_cache, compiler_id: input.compiler)
    Enum.map(analysis.programs, &policy.compile.(&1, options))
  end

  defp link(%{cached: cached}, _, _, _) when not is_nil(cached), do: cached["artifact"]
  defp link(%{backend: "interp"}, _, _, _), do: %{"backend" => "interp"}

  defp link(input, reports, opts, policy) do
    objects = reports |> Enum.flat_map(& &1.objects) |> Enum.uniq() |> Enum.sort()
    library = policy.link.(objects, input.directory, "program-#{input.key}", opts)

    %{
      "backend" => "llvm",
      "library" => Path.basename(library),
      "sha256" => ArtifactCache.digest(File.read!(library)),
      "target" => policy.target.(),
      "symbols" => Enum.map(reports, & &1.symbol)
    }
  end

  defp publish(%{read: input, analyze: analysis, compile: compilation, link: artifact}, output, policy) do
    manifest =
      input.cached ||
        Map.merge(input.metadata, %{
          "identity" => policy.identity_prefix <> input.key,
          "compiler" => input.compiler,
          "source" => input.source,
          "artifact" => artifact,
          "programs" => analysis.programs
        })

    artifacts = if name = artifact["library"], do: [{name, artifact["sha256"]}], else: []
    image = ArtifactCache.publish_bundle!(input, output, manifest, artifacts)

    report = %{
      "identity" => manifest["identity"],
      "backend" => artifact["backend"],
      "image" => image,
      "image_cache_hit" => not is_nil(input.cached),
      "analyzed_programs" => analysis.count,
      "compilation" => Enum.map(compilation, &Map.delete(&1, :objects))
    }

    Map.merge(report, policy.report.(manifest))
  end
end
