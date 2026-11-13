defmodule Beaver.Pattern.PDL do
  @moduledoc """
  Portable PDL collections indexed by explicit callee metadata: `{symbols, source}`, or `:all` for unrestricted rules.
  Selection revisits new callees introduced by rewriting. Compiled native pattern sets remain local to each run.
  """
  alias Beaver.MLIR.{Rewrite, SymbolTable}
  @enforce_keys [:entries, :digest]
  defstruct [:entries, :digest]

  def new(entries) do
    entries =
      Enum.map(entries, fn {symbols, source} when is_binary(source) ->
        symbols = if symbols == :all, do: :all, else: Enum.uniq(List.wrap(symbols))

        unless symbols == :all or (symbols != [] and Enum.all?(symbols, &is_binary/1)),
          do: raise(ArgumentError, "PDL entries require callee symbols or :all")

        {symbols, source}
      end)

    digest = :crypto.hash(:sha256, :erlang.term_to_binary(entries)) |> Base.encode16(case: :lower)
    %__MODULE__{entries: entries, digest: digest}
  end

  def fetch(key, revision, build) when is_function(build, 0),
    do: Beaver.Build.ArtifactCache.fetch_revision(__MODULE__, key, revision, fn -> new(build.()) end)

  def source(%__MODULE__{entries: entries}),
    do: IO.iodata_to_binary(["module {\n", Enum.map(entries, &elem(&1, 1)), "}\n"])

  def applicable?(rules, symbols), do: Enum.any?(rules.entries, &matches?(&1, symbols))
  defp matches?({:all, _}, _), do: true
  defp matches?({roots, _}, symbols), do: Enum.any?(roots, &(&1 in symbols))
  @doc "Applies relevant rules, including rules enabled by symbols introduced during rewriting. Returns module."
  def apply!(%__MODULE__{} = rules, module), do: rewrite(rules.entries, module, MapSet.new())

  defp rewrite(entries, module, selected) do
    symbols = SymbolTable.referenced_symbols(module)

    next =
      Enum.with_index(entries)
      |> Enum.reduce(selected, fn {entry, i}, acc -> if matches?(entry, symbols), do: MapSet.put(acc, i), else: acc end)

    if next == selected do
      module
    else
      sources = for {{_, source}, i} <- Enum.with_index(entries), MapSet.member?(next, i), do: source
      Rewrite.apply_patterns!(module, [{:pdl, IO.iodata_to_binary(["module {\n", sources, "}\n"])}])
      rewrite(entries, module, next)
    end
  end
end
