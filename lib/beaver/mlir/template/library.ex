defmodule Beaver.MLIR.Template.Library do
  @moduledoc """
  Verified templates stored as context-independent bytecode and signatures. `fetch/3` caches one revision per key;
  include all source inputs in that revision. `with_module/3` owns fresh IR whose handles must not escape.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.{Bytecode, Context, Module, SymbolTable}
  @enforce_keys [:bytecode, :signatures, :digest]
  defstruct [:bytecode, :signatures, :digest]

  @doc """
  Defines a provider over named `kind_source/0` callbacks. `:libraries` maps kinds to import visibility;
  `:templates` selects `(symbol, kind)` roots and `:signatures` selects declared symbols.
  `:dependencies` lists generator modules whose revisions invalidate cached bytecode.
  """
  defmacro __using__(opts) do
    libraries = Keyword.fetch!(opts, :libraries)
    select = Keyword.fetch!(opts, :templates)
    signatures = Keyword.fetch!(opts, :signatures)

    quote do
      @template_dependencies [__MODULE__ | unquote(opts[:dependencies] || [])]
      defp library(kind) do
        revision = Enum.map(@template_dependencies, & &1.__info__(:md5))
        unquote(__MODULE__).fetch({__MODULE__, kind}, revision, fn -> apply(__MODULE__, :"#{kind}_source", []) end)
      end

      def declaration_signatures,
        do: Map.filter(library(:inline).signatures, fn {name, _} -> unquote(signatures).(name) end)

      def with_declarations(source),
        do: unquote(__MODULE__).with_declarations(%{signatures: declaration_signatures()}, source)

      def ensure_declarations!(module) do
        symbols = Enum.filter(Beaver.MLIR.SymbolTable.referenced_symbols(module), unquote(signatures))
        unquote(__MODULE__).declare!(library(:inline), module, symbols: symbols, visibility: :private)
      end

      unquote_splicing(
        for {kind, visibility} <- libraries do
          quote do
            def unquote(:"with_#{kind}_module")(ctx, fun),
              do: unquote(__MODULE__).with_module(library(unquote(kind)), ctx, fun)

            def unquote(:"import_#{kind}!")(module, opts \\ []) do
              defaults = [
                visibility: unquote(visibility),
                verify_type: true,
                select: fn name -> unquote(select).(name, unquote(kind)) end
              ]

              unquote(__MODULE__).import!(library(unquote(kind)), module, Keyword.merge(defaults, opts))
            end
          end
        end
      )
    end
  end

  def new(source) when is_binary(source) do
    Context.with_context(fn ctx ->
      Module.with_module(ctx, source, fn module ->
        %__MODULE__{
          bytecode: Bytecode.write!(module),
          signatures: SymbolTable.function_signatures(module),
          digest: :crypto.hash(:sha256, source) |> Base.encode16(case: :lower)
        }
      end)
    end)
  end

  def fetch(key, revision, source) when is_function(source, 0),
    do: Beaver.Build.ArtifactCache.fetch_revision(__MODULE__, key, revision, fn -> new(source.()) end)

  def with_module(%__MODULE__{} = library, ctx, fun), do: Module.with_module(ctx, library.bytecode, fun)
  @doc "Imports selected roots and their dependencies; optional `:select` filters explicit or referenced roots."
  def import!(library, target, opts \\ []) do
    {select, opts} = Keyword.pop(opts, :select)
    symbols = Keyword.get(opts, :symbols, :referenced)
    symbols = if symbols == :referenced, do: SymbolTable.referenced_symbols(target), else: symbols
    symbols = if select, do: Enum.filter(symbols, select), else: symbols
    opts = Keyword.put(opts, :symbols, symbols)

    if symbols == [],
      do: target,
      else: with_module(library, MLIR.context(target), &Module.import_symbols!(target, &1, opts))
  end

  @doc "Adds missing flat function declarations from this library before parsing textual MLIR."
  def with_declarations(library, source) when is_binary(source) do
    declarations =
      for {name, {inputs, outputs}} <- Enum.sort(library.signatures),
          symbol = Beaver.MLIR.Template.check_symbol!(name),
          token = Regex.escape(symbol) <> "(?=[\\s(),{}:]|$)",
          Regex.match?(Regex.compile!(token), source),
          not Regex.match?(Regex.compile!("func\\.func\\s+(?:(?:private|public|nested)\\s+)?" <> token), source) do
        result = if outputs == [], do: "", else: " -> (" <> Enum.join(outputs, ", ") <> ")"
        "  func.func private #{symbol}(#{Enum.join(inputs, ", ")})#{result}"
      end

    insert_declarations(source, Enum.join(declarations, "\n"))
  end

  defp insert_declarations(source, ""), do: source

  defp insert_declarations(source, declarations) do
    opening = ~r/\A(\s*module\b[^{]*\{)/

    if Regex.match?(opening, source),
      do: Regex.replace(opening, source, fn prefix, _ -> prefix <> "\n" <> declarations end, global: false),
      else: declarations <> "\n" <> source
  end

  def declare!(library, target, opts \\ []) do
    symbols = Keyword.get_lazy(opts, :symbols, fn -> SymbolTable.referenced_symbols(target) end)

    signatures =
      for symbol <- Enum.uniq(symbols),
          Map.has_key?(library.signatures, symbol),
          do: {symbol, library.signatures[symbol]}

    SymbolTable.ensure_declarations!(target, signatures, Keyword.delete(opts, :symbols))
  end
end
