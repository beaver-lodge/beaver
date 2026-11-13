defmodule Beaver.Build.ArtifactCache do
  @moduledoc """
  Digest-checked artifacts and atomic publication. JSON caches publish checksums before manifests.
  Incomplete entries are misses; complete entries with bad digests raise. Temporary files are cleaned on failure.
  Publish referenced artifacts before their manifest; renaming a group of files is not a filesystem transaction.
  """
  @doc "Caches portable data by owner/key and revision; callers include every generator dependency."
  def fetch_revision(owner, key, revision, build) do
    key = {owner, key}
    revision = {owner.__info__(:md5), revision}

    case :persistent_term.get(key, nil) do
      {^revision, value} -> value
      _ -> tap(build.(), &:persistent_term.put(key, {revision, &1}))
    end
  end

  def digest(bytes), do: :crypto.hash(:sha256, bytes) |> Base.encode16(case: :lower)
  @doc "Maps JSON inputs through a revisioned analysis cache and returns the number of newly built entries."
  def map_json!(items, directory, revision, ready?, build) do
    Enum.map_reduce(items, 0, fn item, count ->
      if ready?.(item) do
        {item, count}
      else
        path = Path.join(directory, digest([revision, JSON.encode!(item)]))
        {value, hit?} = fetch_json!(path <> ".json", path <> ".sha256", build, [item])
        {value, count + if(hit?, do: 0, else: 1)}
      end
    end)
  end

  @doc "Opens a manifest bundle; a present manifest must have a valid checksum."
  def bundle(directory, manifest_name, checksum_name) do
    manifest = Path.join(directory, manifest_name)
    checksum = Path.join(directory, checksum_name)

    %{
      directory: directory,
      manifest: manifest,
      checksum: checksum,
      cached: if(File.regular?(manifest), do: read_json!(manifest, checksum))
    }
  end

  @doc "Verifies a bundle and copies `{relative_path, digest}` artifacts before publishing its manifest."
  def publish_bundle!(bundle, output, manifest, artifacts \\ []) do
    if is_nil(bundle.cached), do: write_json!(bundle.manifest, bundle.checksum, manifest)
    bytes = read_verified!(bundle.manifest, File.read!(bundle.checksum))

    for {name, digest} <- artifacts,
        do: copy_verified!(Path.join(bundle.directory, name), Path.join(output, name), digest)

    write_atomic!(Path.join(output, Path.basename(bundle.checksum)), digest(bytes))
    write_atomic!(Path.join(output, Path.basename(bundle.manifest)), bytes)
  end

  def read_verified!(path, expected) do
    bytes = File.read!(path)
    unless digest(bytes) == expected, do: raise(ArgumentError, "cache digest mismatch: #{path}")
    bytes
  end

  def read_json!(path, checksum), do: path |> read_verified!(File.read!(checksum)) |> JSON.decode!()

  def fetch_json!(path, checksum, build, args \\ []) when is_function(build, length(args)) do
    if File.regular?(path) and File.regular?(checksum) do
      {read_json!(path, checksum), true}
    else
      {write_json!(path, checksum, apply(build, args)), false}
    end
  end

  def write_json!(path, checksum, value) do
    bytes = JSON.encode!(value)
    write_atomic!(checksum, digest(bytes))
    write_atomic!(path, bytes)
    value
  end

  def copy_verified!(source, destination, expected), do: write_atomic!(destination, read_verified!(source, expected))

  def write_atomic!(path, data) do
    bytes = IO.iodata_to_binary(data)

    unless File.read(path) == {:ok, bytes} do
      File.mkdir_p!(Path.dirname(path))
      temporary = "#{path}.tmp-#{System.pid()}-#{System.unique_integer([:positive])}"

      try do
        File.write!(temporary, bytes)
        File.rename!(temporary, path)
      after
        File.rm(temporary)
      end
    end

    path
  end
end
