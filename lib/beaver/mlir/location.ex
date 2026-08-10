defmodule Beaver.MLIR.Location do
  @moduledoc """
  Constructors and safe inspection helpers for MLIR locations.

  Locations are owned and uniqued by an MLIR context. In addition to source
  ranges, MLIR can represent named provenance, call sites, and fused
  provenance with an arbitrary metadata attribute. `fused/2` accepts locations
  as well as location-bearing operations and values so transformations can
  preserve provenance without dropping to the C API.

  MLIR may canonicalize a fused location without metadata to its only child.
  Use `with_metadata/3` when the metadata itself must be retained even for a
  single source location.
  """
  alias Beaver.Deferred
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  @type source() ::
          t()
          | MLIR.Operation.t()
          | MLIR.Value.t()
          | Deferred.contextual(t() | MLIR.Operation.t() | MLIR.Value.t())

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

    Deferred.from_opts(
      opts,
      &CAPI.mlirLocationFileLineColGet(&1, MLIR.StringRef.create(name), line, column)
    )
  end

  @doc """
  Creates a source range location.

  Unlike `file/1`, all range coordinates are explicit so an accidentally
  truncated range cannot silently be constructed.
  """
  @spec file_range(file_range_opts()) :: Deferred.contextual(t())
  def file_range(opts) when is_list(opts) do
    name = Keyword.fetch!(opts, :name)
    start_line = Keyword.fetch!(opts, :start_line)
    start_column = Keyword.fetch!(opts, :start_column)
    end_line = Keyword.fetch!(opts, :end_line)
    end_column = Keyword.fetch!(opts, :end_column)

    Deferred.from_opts(opts, fn ctx ->
      CAPI.mlirLocationFileLineColRangeGet(
        ctx,
        MLIR.StringRef.create(name),
        start_line,
        start_column,
        end_line,
        end_column
      )
    end)
  end

  @doc "Creates an MLIR location from `Macro.Env`-like metadata."
  @spec from_env(env_like(), env_opts()) :: Deferred.contextual(t())
  def from_env(env, opts \\ []) when is_map(env) do
    name = env |> Map.get(:file, "nofile") |> to_string()
    line = Map.get(env, :line)
    line = if is_integer(line), do: line, else: 0

    file(env_file_opts(name, line, opts))
  end

  @doc "Creates an unknown location."
  @spec unknown(Deferred.opts()) :: Deferred.contextual(t())
  def unknown(opts \\ []) do
    Deferred.from_opts(opts, &CAPI.mlirLocationUnknownGet/1)
  end

  @doc """
  Creates a named location with an unknown child.

  Pass an operation, value, or location as the second argument to retain its
  existing provenance as the child.
  """
  @spec named(String.Chars.t(), Deferred.opts()) :: Deferred.contextual(t())
  def named(name, opts) when is_list(opts) do
    Deferred.from_opts(opts, fn ctx ->
      CAPI.mlirLocationNameGet(
        ctx,
        MLIR.StringRef.create(name),
        CAPI.mlirLocationUnknownGet(ctx)
      )
    end)
  end

  @spec named(String.Chars.t(), source(), Deferred.opts()) :: Deferred.contextual(t())
  def named(name, child, opts \\ []) do
    opts = infer_context(opts, [child])

    Deferred.from_opts(opts, fn ctx ->
      CAPI.mlirLocationNameGet(
        ctx,
        MLIR.StringRef.create(name),
        resolve(child, ctx)
      )
    end)
  end

  @doc """
  Creates a call-site location from callee and caller provenance.

  Each argument may be a location, operation, value, or deferred equivalent.
  """
  @spec call_site(source(), source(), Deferred.opts()) :: Deferred.contextual(t())
  def call_site(callee, caller, opts \\ []) do
    opts = infer_context(opts, [callee, caller])

    Deferred.from_opts(opts, fn ctx ->
      CAPI.mlirLocationCallSiteGet(resolve(callee, ctx), resolve(caller, ctx))
    end)
  end

  @doc """
  Fuses location-bearing sources, optionally retaining an MLIR attribute as
  metadata.

  `sources` may contain locations, operations, values, or deferred
  equivalents. MLIR removes unknown children, deduplicates children, and may
  return a non-fused location when no metadata is supplied and only one unique
  child remains.
  """
  @spec fused([source()], keyword()) :: Deferred.contextual(t())
  def fused(sources, opts \\ []) when is_list(sources) do
    metadata = Keyword.get(opts, :metadata)
    opts = infer_context(opts, sources ++ List.wrap(metadata))

    Deferred.from_opts(opts, fn ctx ->
      locations = Enum.map(sources, &resolve(&1, ctx))
      metadata = resolve_metadata(metadata, ctx)

      CAPI.mlirLocationFusedGet(
        ctx,
        length(locations),
        Beaver.Native.array(locations, __MODULE__),
        metadata
      )
    end)
  end

  @doc """
  Attaches metadata while retaining all supplied source provenance.

  This is the explicit metadata-preserving form of `fused/2` and is useful for
  transformation tags, debug scopes, and other provenance attributes.
  """
  @spec with_metadata(source() | [source()], MLIR.Attribute.t() | Deferred.attribute(), keyword()) ::
          Deferred.contextual(t())
  def with_metadata(sources, metadata, opts \\ []) do
    fused(List.wrap(sources), Keyword.put(opts, :metadata, metadata))
  end

  @doc "Returns the location carried by a location, operation, or value."
  @spec of(source()) :: Deferred.contextual(t())
  def of(%__MODULE__{} = location), do: location
  def of(%MLIR.Operation{} = operation), do: MLIR.Operation.location(operation)
  def of(%MLIR.Value{} = value), do: MLIR.Value.location(value)

  def of(%Deferred{} = deferred) do
    Deferred.defer(fn ctx -> deferred |> Deferred.resolve(ctx) |> of() end)
  end

  def of(other) do
    raise ArgumentError,
          "expected an MLIR location, operation, value, or deferred equivalent, got: #{inspect(other)}"
  end

  @doc "Resolves a location-bearing value in `ctx` and verifies context ownership."
  @spec resolve(source(), MLIR.Context.t()) :: t()
  def resolve(source, %MLIR.Context{} = ctx) do
    location = source |> Deferred.resolve(ctx) |> of()

    MLIR.Context.ensure_same!(location, ctx)
  end

  @doc "Returns the built-in location kind, or `:other` for dialect-defined locations."
  @spec kind(t()) :: kind()
  def kind(%__MODULE__{} = location) do
    cond do
      unknown?(location) -> :unknown
      file_range?(location) -> :file_range
      name?(location) -> :name
      call_site?(location) -> :call_site
      fused?(location) -> :fused
      true -> :other
    end
  end

  @doc "Returns structured source coordinates for a file range location."
  @spec source_range(t()) :: {:ok, source_range()} | :error
  def source_range(%__MODULE__{} = location) do
    if file_range?(location) do
      {:ok,
       %{
         file: CAPI.mlirLocationFileLineColRangeGetFilename(location) |> to_string(),
         start_line: location |> CAPI.mlirLocationFileLineColRangeGetStartLine() |> to_term(),
         start_column: location |> CAPI.mlirLocationFileLineColRangeGetStartColumn() |> to_term(),
         end_line: location |> CAPI.mlirLocationFileLineColRangeGetEndLine() |> to_term(),
         end_column: location |> CAPI.mlirLocationFileLineColRangeGetEndColumn() |> to_term()
       }}
    else
      :error
    end
  end

  @doc "Returns the label of a named location."
  @spec name(t()) :: {:ok, String.t()} | :error
  def name(%__MODULE__{} = location) do
    if name?(location) do
      {:ok, location |> CAPI.mlirLocationNameGetName() |> to_string()}
    else
      :error
    end
  end

  @doc "Returns the child provenance of a named location."
  @spec child(t()) :: {:ok, t()} | :error
  def child(%__MODULE__{} = location) do
    if name?(location), do: {:ok, CAPI.mlirLocationNameGetChildLoc(location)}, else: :error
  end

  @doc "Returns the callee provenance of a call-site location."
  @spec callee(t()) :: {:ok, t()} | :error
  def callee(%__MODULE__{} = location) do
    if call_site?(location), do: {:ok, CAPI.mlirLocationCallSiteGetCallee(location)}, else: :error
  end

  @doc "Returns the caller provenance of a call-site location."
  @spec caller(t()) :: {:ok, t()} | :error
  def caller(%__MODULE__{} = location) do
    if call_site?(location), do: {:ok, CAPI.mlirLocationCallSiteGetCaller(location)}, else: :error
  end

  @doc "Returns the number of source locations retained by a fused location."
  @spec location_count(t()) :: {:ok, non_neg_integer()} | :error
  def location_count(%__MODULE__{} = location) do
    if fused?(location) do
      {:ok, location |> CAPI.mlirLocationFusedGetNumLocations() |> to_term()}
    else
      :error
    end
  end

  @doc "Returns the source locations retained by a fused location."
  @spec locations(t()) :: {:ok, [t()]} | :error
  def locations(%__MODULE__{} = location) do
    with true <- fused?(location),
         {:ok, count} <- location_count(location) do
      {:ok,
       for position <- zero_based_range(count) do
         CAPI.beaverLocationFusedGetLocationAt(location, position)
       end}
    else
      _ -> :error
    end
  end

  @doc """
  Returns a fused location's metadata attribute.

  The returned attribute may be MLIR's null attribute when the fused location
  has no metadata; use `Beaver.MLIR.null?/1` to distinguish that case.
  """
  @spec metadata(t()) :: {:ok, MLIR.Attribute.t()} | :error
  def metadata(%__MODULE__{} = location) do
    if fused?(location), do: {:ok, CAPI.mlirLocationFusedGetMetadata(location)}, else: :error
  end

  @doc "Returns the location as its underlying location attribute."
  @spec attribute(t()) :: MLIR.Attribute.t()
  def attribute(%__MODULE__{} = location), do: CAPI.mlirLocationGetAttribute(location)

  @doc "Converts a location attribute back to a location."
  @spec from_attribute(MLIR.Attribute.t()) :: t()
  def from_attribute(%MLIR.Attribute{} = attribute) do
    unless MLIR.Attribute.location?(attribute) do
      raise ArgumentError, "expected an MLIR location attribute"
    end

    CAPI.mlirLocationFromAttribute(attribute)
  end

  @doc "Returns whether the location is an MLIR unknown location."
  @spec unknown?(t()) :: boolean()
  def unknown?(%__MODULE__{} = location),
    do: location |> CAPI.mlirLocationIsAUnknown() |> to_term()

  @doc "Returns whether the location is an MLIR file range location."
  @spec file_range?(t()) :: boolean()
  def file_range?(%__MODULE__{} = location),
    do: location |> CAPI.mlirLocationIsAFileLineColRange() |> to_term()

  @doc "Returns whether the location is an MLIR named location."
  @spec name?(t()) :: boolean()
  def name?(%__MODULE__{} = location),
    do: location |> CAPI.mlirLocationIsAName() |> to_term()

  @doc "Returns whether the location is an MLIR call-site location."
  @spec call_site?(t()) :: boolean()
  def call_site?(%__MODULE__{} = location),
    do: location |> CAPI.mlirLocationIsACallSite() |> to_term()

  @doc "Returns whether the location is an MLIR fused location."
  @spec fused?(t()) :: boolean()
  def fused?(%__MODULE__{} = location),
    do: location |> CAPI.mlirLocationIsAFused() |> to_term()

  defp resolve_metadata(nil, _ctx), do: MLIR.Attribute.null()

  defp resolve_metadata(metadata, ctx) do
    case Deferred.resolve(metadata, ctx) do
      %MLIR.Attribute{} = attribute ->
        MLIR.Context.ensure_same!(attribute, ctx)

      other ->
        raise ArgumentError, "location metadata must be an MLIR attribute, got: #{inspect(other)}"
    end
  end

  defp infer_context(opts, sources) do
    case Deferred.context(opts) || Enum.find_value(sources, &context_of/1) do
      nil -> opts
      ctx -> Keyword.put(opts, :ctx, ctx)
    end
  end

  defp context_of(%module{} = entity)
       when module in [__MODULE__, MLIR.Operation, MLIR.Value, MLIR.Attribute],
       do: MLIR.context(entity)

  defp context_of(_other), do: nil

  defp zero_based_range(0), do: []
  defp zero_based_range(count), do: 0..(count - 1)

  defp to_term(value), do: Beaver.Native.to_term(value)

  @spec env_file_opts(String.t(), integer(), env_opts()) :: file_opts()
  defp env_file_opts(name, line, opts) do
    opts
    |> Keyword.put(:name, name)
    |> Keyword.put(:line, line)
  end
end
