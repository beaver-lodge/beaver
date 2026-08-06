defmodule Beaver.MLIR.CompilationCache.File do
  @moduledoc """
  Filesystem backend for incremental compilation artifacts.

  Entries are written through a temporary file and atomically renamed. Reads
  use safe Erlang term decoding; malformed entries are reported to the runtime,
  which invalidates them and recompiles.
  """

  defstruct [:root]

  @type t :: %__MODULE__{root: Path.t()}

  @spec new(Path.t()) :: t()
  def new(root), do: %__MODULE__{root: Path.expand(root)}

  @spec get(t(), String.t()) :: {:ok, map()} | :miss | {:error, term()}
  def get(cache, key) do
    path = path(cache, key)

    case File.read(path) do
      {:ok, binary} ->
        try do
          case :erlang.binary_to_term(binary, [:safe]) do
            value when is_map(value) -> {:ok, value}
            _ -> {:error, :invalid_entry}
          end
        rescue
          ArgumentError -> {:error, :invalid_entry}
        end

      {:error, :enoent} ->
        :miss

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec put(t(), String.t(), map()) :: :ok | {:error, term()}
  def put(cache, key, value) do
    with :ok <- File.mkdir_p(cache.root) do
      destination = path(cache, key)
      temporary = destination <> ".tmp.#{System.unique_integer([:positive])}"

      try do
        case File.write(temporary, :erlang.term_to_binary(value, [:deterministic])) do
          :ok -> File.rename(temporary, destination)
          error -> error
        end
      after
        File.rm(temporary)
      end
    end
  end

  @spec delete(t(), String.t()) :: :ok | {:error, term()}
  def delete(cache, key) do
    case File.rm(path(cache, key)) do
      :ok -> :ok
      {:error, :enoent} -> :ok
      other -> other
    end
  end

  @spec clear(t()) :: :ok | {:error, term()}
  def clear(cache) do
    with {:ok, entries} <- list(cache) do
      Enum.reduce_while(entries, :ok, &remove_entry(cache.root, &1, &2))
    end
  end

  defp remove_entry(root, entry, :ok) do
    case File.rm(Path.join(root, entry)) do
      :ok -> {:cont, :ok}
      {:error, :enoent} -> {:cont, :ok}
      error -> {:halt, error}
    end
  end

  defp list(cache) do
    case File.ls(cache.root) do
      {:ok, entries} -> {:ok, Enum.filter(entries, &String.ends_with?(&1, ".beaver-cache"))}
      {:error, :enoent} -> {:ok, []}
      other -> other
    end
  end

  defp path(cache, key), do: Path.join(cache.root, key <> ".beaver-cache")
end
