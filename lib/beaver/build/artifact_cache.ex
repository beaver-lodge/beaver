defmodule Beaver.Build.ArtifactCache do
  @moduledoc """
  Digest-checked artifacts and atomic single-file publication.

  JSON cache entries publish their checksum before their manifest. Incomplete
  entries are misses; a complete entry with a bad digest raises instead of silently
  trusting corrupted data. Object caches may deliberately treat that as a miss.
  Files are written beside their destination and renamed, with temporary files
  cleaned on failure. Callers publish referenced artifacts before their manifest.
  """

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

  def copy_verified!(source, destination, expected) do
    bytes = read_verified!(source, expected)
    write_atomic!(destination, bytes)
  end

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
