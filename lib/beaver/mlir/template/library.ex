defmodule Beaver.MLIR.Template.Library do
  @moduledoc """
  Verified MLIR templates stored as context-independent bytecode and textual function signatures.

  `fetch/3` caches one revision per caller key. The revision must include all inputs used by the source builder.
  Each `with_module/3` call owns a fresh module in the supplied context; its native handles must not escape.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.{Bytecode, Context, Module, SymbolTable}
  @enforce_keys [:bytecode, :signatures, :digest]
  defstruct [:bytecode, :signatures, :digest]

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

  def fetch(key, revision, source) when is_function(source, 0) do
    key = {__MODULE__, key}
    revision = {__MODULE__.__info__(:md5), revision}

    case :persistent_term.get(key, nil) do
      {^revision, library} -> library
      _ -> tap(new(source.()), &:persistent_term.put(key, {revision, &1}))
    end
  end

  def with_module(%__MODULE__{} = library, ctx, fun), do: Module.with_module(ctx, library.bytecode, fun)

  @doc "Imports selected roots and their dependencies; optional `:select` filters explicit or referenced roots."
  def import!(library, target, opts \\ []) do
    case Keyword.pop(opts, :select) do
      {nil, opts} -> with_module(library, MLIR.context(target), &Module.import_symbols!(target, &1, opts))
      {select, opts} -> import_selected!(library, target, select, opts)
    end
  end

  defp import_selected!(library, target, select, opts) do
    symbols = Keyword.get_lazy(opts, :symbols, fn -> SymbolTable.referenced_symbols(target) end)
    symbols = if symbols == :referenced, do: SymbolTable.referenced_symbols(target), else: symbols
    selected = Enum.filter(symbols, select)

    if selected == [],
      do: target,
      else:
        with_module(
          library,
          MLIR.context(target),
          &Module.import_symbols!(target, &1, Keyword.put(opts, :symbols, selected))
        )
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
