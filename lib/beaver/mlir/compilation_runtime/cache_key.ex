defmodule Beaver.MLIR.CompilationRuntime.CacheKey do
  @moduledoc "Deterministic compatibility keys for incremental MLIR artifacts."

  @version 1

  @spec lookup(map()) :: String.t()
  def lookup(inputs), do: digest({:beaver_mlir_lookup, @version, normalize(inputs)})

  @spec artifact(map(), non_neg_integer()) :: String.t()
  def artifact(inputs, structural_hash) do
    digest({:beaver_mlir_artifact, @version, normalize(inputs), structural_hash})
  end

  @spec digest(term()) :: String.t()
  def digest(term) do
    term
    |> :erlang.term_to_binary([:deterministic])
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
  end

  defp normalize(%_{} = struct) do
    {:struct, struct.__struct__ |> Atom.to_string(), struct |> Map.from_struct() |> normalize()}
  end

  defp normalize(map) when is_map(map) do
    {:map,
     map
     |> Enum.map(fn {key, value} -> {normalize(key), normalize(value)} end)
     |> Enum.sort()}
  end

  defp normalize(tuple) when is_tuple(tuple) do
    {:tuple, tuple |> Tuple.to_list() |> Enum.map(&normalize/1)}
  end

  defp normalize(list) when is_list(list), do: Enum.map(list, &normalize/1)

  defp normalize(value)
       when is_atom(value) or is_binary(value) or is_number(value),
       do: value

  defp normalize(value) do
    raise ArgumentError,
          "cache inputs must be deterministic data, got: #{inspect(value, limit: 5)}"
  end
end
