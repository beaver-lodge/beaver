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

  def import!(library, target, opts \\ []) do
    with_module(library, MLIR.context(target), &Module.import_symbols!(target, &1, opts))
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
