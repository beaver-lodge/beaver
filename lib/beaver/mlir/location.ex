alias Beaver.MLIR.{Attribute, Context, Operation, StringRef, Value}
alias Beaver.Native

defmodule Beaver.MLIR.Location do
  @moduledoc """
  Constructors and safe inspection helpers for MLIR locations.

  Locations belong to a context and carry source ranges, names, call sites or fused provenance. `fused/2` accepts
  locations, operations and values. A fused location without metadata may canonicalize to its only child; use
  `with_metadata/3` to retain metadata even for a single source.
  """
  alias Beaver.Deferred
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Native

  @type source() :: t() | Operation.t() | Value.t() | Deferred.contextual(t() | Operation.t() | Value.t())

  @type file_opts() :: [
          name: String.t(),
          line: non_neg_integer(),
          column: non_neg_integer() | nil,
          ctx: Deferred.context_arg()
        ]
  @type file_range_opts() :: [
          name: String.t(),
          start_line: non_neg_integer(),
          start_column: non_neg_integer(),
          end_line: non_neg_integer(),
          end_column: non_neg_integer(),
          ctx: Deferred.context_arg()
        ]
  @type env_opts() :: [column: non_neg_integer() | nil, ctx: Deferred.context_arg()]
  @type env_like() :: %{optional(:file) => String.t() | nil, optional(:line) => integer() | nil}
  @type source_range() :: %{
          file: String.t(),
          start_line: non_neg_integer(),
          start_column: non_neg_integer(),
          end_line: non_neg_integer(),
          end_column: non_neg_integer()
        }
  @type kind() :: :unknown | :file_range | :name | :call_site | :fused | :other

  @doc """
  Creates a file, line, and column location.

  The column defaults to zero because Elixir metadata usually omits it.

  ## Examples

      iex> ctx = MLIR.Context.create()
      iex> MLIR.Location.file(name: "filename", line: 1, column: 1, ctx: ctx) |> MLIR.to_string()
      ~s{filename:1:1}
      iex> MLIR.Context.destroy(ctx)
  """
  @spec file(file_opts()) :: Deferred.contextual(t())
  def file(opts) do
    name = Keyword.fetch!(opts, :name)
    line = Keyword.fetch!(opts, :line)
    column = Keyword.get(opts, :column) || 0
    Deferred.from_opts(opts, &CAPI.mlirLocationFileLineColGet(&1, StringRef.create(name), line, column))
  end

  @doc "Creates a source range with all coordinates required, preventing silent truncation."
  @spec file_range(file_range_opts()) :: Deferred.contextual(t())
  def file_range(opts) when is_list(opts) do
    name = Keyword.fetch!(opts, :name)
    start_line = Keyword.fetch!(opts, :start_line)
    start_column = Keyword.fetch!(opts, :start_column)
    end_line = Keyword.fetch!(opts, :end_line)
    end_column = Keyword.fetch!(opts, :end_column)

    Deferred.from_opts(opts, fn ctx ->
      CAPI.mlirLocationFileLineColRangeGet(ctx, StringRef.create(name), start_line, start_column, end_line, end_column)
    end)
  end

  @doc "Creates an MLIR location from `Macro.Env`-like metadata."
  @spec from_env(env_like(), env_opts()) :: Deferred.contextual(t())
  def from_env(env, opts \\ []) when is_map(env) do
    name = env |> Map.get(:file, "nofile") |> to_string()
    line = Map.get(env, :line)
    line = if is_integer(line), do: line, else: 0
    file(opts |> Keyword.put(:name, name) |> Keyword.put(:line, line))
  end

  @doc "Creates an unknown location."
  @spec unknown(Deferred.opts()) :: Deferred.contextual(t())
  def unknown(opts \\ []), do: Deferred.from_opts(opts, &CAPI.mlirLocationUnknownGet/1)

  @doc """
  Creates a named location with an unknown child, or retains the provenance of an operation, value or location
  supplied as the second argument.
  """
  @spec named(String.Chars.t(), Deferred.opts()) :: Deferred.contextual(t())
  def named(name, opts) when is_list(opts) do
    Deferred.from_opts(opts, fn ctx ->
      CAPI.mlirLocationNameGet(ctx, StringRef.create(name), CAPI.mlirLocationUnknownGet(ctx))
    end)
  end

  @spec named(String.Chars.t(), source(), Deferred.opts()) :: Deferred.contextual(t())
  def named(name, child, opts \\ []) do
    opts = infer_context(opts, [child])
    Deferred.from_opts(opts, fn ctx -> CAPI.mlirLocationNameGet(ctx, StringRef.create(name), resolve(child, ctx)) end)
  end

  @doc "Creates a call site from callee and caller locations, operations, values or deferred equivalents."
  @spec call_site(source(), source(), Deferred.opts()) :: Deferred.contextual(t())
  def call_site(callee, caller, opts \\ []) do
    opts = infer_context(opts, [callee, caller])
    Deferred.from_opts(opts, fn ctx -> CAPI.mlirLocationCallSiteGet(resolve(callee, ctx), resolve(caller, ctx)) end)
  end

  @doc """
  Fuses locations, operations, values or deferred equivalents with optional `:metadata`. MLIR removes unknown and
  duplicate children; without metadata, a single remaining child may be returned directly.
  """
  @spec fused([source()], keyword()) :: Deferred.contextual(t())
  def fused(sources, opts \\ []) when is_list(sources) do
    metadata = Keyword.get(opts, :metadata)
    opts = infer_context(opts, sources ++ List.wrap(metadata))

    Deferred.from_opts(opts, fn ctx ->
      locations = Enum.map(sources, &resolve(&1, ctx))
      metadata = resolve_metadata(metadata, ctx)
      CAPI.mlirLocationFusedGet(ctx, length(locations), Native.array(locations, __MODULE__), metadata)
    end)
  end

  @doc "Attaches metadata while retaining source provenance; delegates to `fused/2`."
  @spec with_metadata(source() | [source()], Attribute.t() | Deferred.attribute(), keyword()) ::
          Deferred.contextual(t())
  def with_metadata(sources, metadata, opts \\ []),
    do: fused(List.wrap(sources), Keyword.put(opts, :metadata, metadata))

  @doc "Returns the location carried by a location, operation, or value."
  @spec of(source()) :: Deferred.contextual(t())
  def of(%__MODULE__{} = loc), do: loc
  def of(%Operation{} = op), do: Operation.location(op)
  def of(%Value{} = value), do: Value.location(value)

  def of(%Deferred{} = deferred), do: Deferred.defer(fn ctx -> deferred |> Deferred.resolve(ctx) |> of() end)

  def of(other) do
    raise ArgumentError, "expected an MLIR location, operation, value, or deferred equivalent, got: #{inspect(other)}"
  end

  @doc "Resolves a location-bearing value in `ctx` and verifies context ownership."
  @spec resolve(source(), Context.t()) :: t()
  def resolve(source, %Context{} = ctx) do
    location = source |> Deferred.resolve(ctx) |> of()
    Context.ensure_same!(location, ctx)
  end

  @doc "Returns the built-in location kind, or `:other` for dialect-defined locations."
  @spec kind(t()) :: kind()
  def kind(%__MODULE__{} = loc) do
    cond do
      unknown?(loc) -> :unknown
      file_range?(loc) -> :file_range
      name?(loc) -> :name
      call_site?(loc) -> :call_site
      fused?(loc) -> :fused
      true -> :other
    end
  end

  @doc "Returns structured source coordinates for a file range location."
  @spec source_range(t()) :: {:ok, source_range()} | :error
  def source_range(%__MODULE__{} = loc) do
    if file_range?(loc),
      do:
        {:ok,
         %{
           file: CAPI.mlirLocationFileLineColRangeGetFilename(loc) |> to_string(),
           start_line: loc |> CAPI.mlirLocationFileLineColRangeGetStartLine() |> Native.to_term(),
           start_column: loc |> CAPI.mlirLocationFileLineColRangeGetStartColumn() |> Native.to_term(),
           end_line: loc |> CAPI.mlirLocationFileLineColRangeGetEndLine() |> Native.to_term(),
           end_column: loc |> CAPI.mlirLocationFileLineColRangeGetEndColumn() |> Native.to_term()
         }},
      else: :error
  end

  @doc "Returns the label of a named location."
  @spec name(t()) :: {:ok, String.t()} | :error
  def name(%__MODULE__{} = loc),
    do: if(name?(loc), do: {:ok, loc |> CAPI.mlirLocationNameGetName() |> to_string()}, else: :error)

  for {name, predicate, getter, type, doc} <- [
        {:child, :name?, :mlirLocationNameGetChildLoc, quote(do: t()), "Returns a named location's child provenance."},
        {:callee, :call_site?, :mlirLocationCallSiteGetCallee, quote(do: t()),
         "Returns a call site's callee provenance."},
        {:caller, :call_site?, :mlirLocationCallSiteGetCaller, quote(do: t()),
         "Returns a call site's caller provenance."},
        {:metadata, :fused?, :mlirLocationFusedGetMetadata, quote(do: Attribute.t()),
         "Returns fused metadata; use `Beaver.MLIR.null?/1` to check for an absent attribute."}
      ] do
    @doc doc
    @spec unquote(name)(t()) :: {:ok, unquote(type)} | :error
    def unquote(name)(%__MODULE__{} = loc),
      do: if(unquote(predicate)(loc), do: {:ok, CAPI.unquote(getter)(loc)}, else: :error)
  end

  @doc "Returns the number of source locations retained by a fused location."
  @spec location_count(t()) :: {:ok, non_neg_integer()} | :error
  def location_count(%__MODULE__{} = loc),
    do: if(fused?(loc), do: {:ok, loc |> CAPI.mlirLocationFusedGetNumLocations() |> Native.to_term()}, else: :error)

  @doc "Returns the source locations retained by a fused location."
  @spec locations(t()) :: {:ok, [t()]} | :error
  def locations(%__MODULE__{} = loc) do
    with true <- fused?(loc), {:ok, count} <- location_count(loc) do
      {:ok,
       for position <- 0..(count - 1)//1 do
         CAPI.beaverLocationFusedGetLocationAt(loc, position)
       end}
    else
      _ -> :error
    end
  end

  @doc "Returns the location as its underlying location attribute."
  @spec attribute(t()) :: Attribute.t()
  def attribute(%__MODULE__{} = loc), do: CAPI.mlirLocationGetAttribute(loc)

  @doc "Converts a location attribute back to a location."
  @spec from_attribute(Attribute.t()) :: t()
  def from_attribute(%Attribute{} = attribute) do
    unless Attribute.location?(attribute), do: raise(ArgumentError, "expected an MLIR location attribute")

    CAPI.mlirLocationFromAttribute(attribute)
  end

  for {name, suffix} <- [
        unknown: "Unknown",
        file_range: "FileLineColRange",
        name: "Name",
        call_site: "CallSite",
        fused: "Fused"
      ] do
    @doc "Returns whether the location is an MLIR #{name} location."
    @spec unquote(:"#{name}?")(t()) :: boolean()
    def unquote(:"#{name}?")(%__MODULE__{} = loc),
      do: CAPI.unquote(:"mlirLocationIsA#{suffix}")(loc) |> Native.to_term()
  end

  defp resolve_metadata(nil, _ctx), do: Attribute.null()

  defp resolve_metadata(metadata, ctx) do
    case Deferred.resolve(metadata, ctx) do
      %Attribute{} = attribute -> Context.ensure_same!(attribute, ctx)
      other -> raise ArgumentError, "location metadata must be an MLIR attribute, got: #{inspect(other)}"
    end
  end

  defp infer_context(opts, sources) do
    case Deferred.context(opts) || Enum.find_value(sources, &context_of/1) do
      nil -> opts
      ctx -> Keyword.put(opts, :ctx, ctx)
    end
  end

  defp context_of(%module{} = entity) when module in [__MODULE__, Operation, Value, Attribute], do: MLIR.context(entity)

  defp context_of(_other), do: nil
end
