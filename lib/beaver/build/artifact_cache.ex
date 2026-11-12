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

  def read_verified!(path, expected) do
    bytes = File.read!(path)
    unless digest(bytes) == expected, do: raise(ArgumentError, "cache digest mismatch: #{path}")
    bytes
  end

  def read_json!(path, checksum), do: path |> read_verified!(File.read!(checksum)) |> JSON.decode!()

  def fetch_json!(path, checksum, build) when is_function(build, 0) do
    if File.regular?(path) and File.regular?(checksum) do
      {read_json!(path, checksum), true}
    else
      value = build.()
      write_json!(path, checksum, value)
      {value, false}
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
