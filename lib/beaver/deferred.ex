defmodule Beaver.Deferred do
  @moduledoc """
  Explicit, context-bound construction of MLIR entities.

  MLIR types, attributes, locations, and similar entities belong to an
  `Beaver.MLIR.Context`. Builders return a `%Beaver.Deferred{}` when no `:ctx`
  option is supplied and return the concrete entity when one is supplied.

  A deferred value is deliberately distinct from an ordinary one-argument
  callback. Materialize it with `resolve/2`; passing a concrete context-owned
  entity through `resolve/2` also verifies that it belongs to that context.
  """
  alias Beaver.MLIR

  @enforce_keys [:resolver]
  defstruct [:resolver]

  @type context_arg() :: MLIR.Context.t()
  @opaque t(value) :: %__MODULE__{resolver: (context_arg() -> value)}
  @type contextual(value) :: value | t(value)
  @type opts :: [ctx: context_arg()]
  @type type :: contextual(MLIR.Type.t())
  @type operation :: contextual(MLIR.Operation.t())
  @type attribute :: contextual(MLIR.Attribute.t())

  @doc "Wraps a context resolver as an explicit deferred value."
  @spec defer((context_arg() -> value)) :: t(value) when value: var
  def defer(resolver) when is_function(resolver, 1), do: %__MODULE__{resolver: resolver}

  @doc """
  Runs `resolver` immediately when `opts` contains `:ctx`; otherwise defers it.

  An explicitly supplied `ctx: nil` or any non-context value is rejected rather
  than being treated as if the option were absent.
  """
  @spec from_opts(opts(), (context_arg() -> value)) :: contextual(value) when value: var
  def from_opts(opts, resolver) when is_list(opts) and is_function(resolver, 1) do
    case Keyword.fetch(opts, :ctx) do
      :error ->
        defer(resolver)

      {:ok, %MLIR.Context{} = ctx} ->
        resolver.(ctx) |> MLIR.Context.ensure_same!(ctx)

      {:ok, invalid} ->
        raise ArgumentError, "expected :ctx to be an MLIR context, got: #{inspect(invalid)}"
    end
  end

  @doc "Returns the optional context in `opts`, validating an explicitly supplied value."
  @spec context(opts()) :: context_arg() | nil
  def context(opts) when is_list(opts) do
    case Keyword.fetch(opts, :ctx) do
      :error ->
        nil

      {:ok, %MLIR.Context{} = ctx} ->
        ctx

      {:ok, invalid} ->
        raise ArgumentError, "expected :ctx to be an MLIR context, got: #{inspect(invalid)}"
    end
  end

  @spec fetch_insertion_point(keyword()) ::
          MLIR.Block.t() | MLIR.PatternRewriter.t() | MLIR.RewriterBase.t() | Macro.t() | nil
  def fetch_insertion_point(opts) do
    for key <- [:block, :blk] do
      if opts[key] do
        raise ArgumentError, "use :ip instead of :#{key} as key"
      end
    end

    opts[:ip]
  end

  @doc """
  Materializes a deferred value in `ctx`.

  Concrete context-owned entities pass through unchanged after their context
  ownership has been checked. Resolvers may return `{:ok, value}` or
  `{:error, reason}`; success is unwrapped and errors become `ArgumentError`.
  """
  @spec resolve(contextual(value), context_arg()) :: value when value: var
  def resolve(resolver, %MLIR.Context{}) when is_function(resolver, 1) do
    raise ArgumentError,
          "bare context resolvers are not deferred values; wrap the function with defer/1"
  end

  def resolve(value, %MLIR.Context{} = ctx) do
    value
    |> do_resolve(ctx)
    |> MLIR.Context.ensure_same!(ctx)
  end

  def resolve(_value, invalid) do
    raise ArgumentError, "expected an MLIR context, got: #{inspect(invalid)}"
  end

  defp do_resolve({:parametric, _, _, deferred}, ctx), do: do_resolve(deferred, ctx)

  defp do_resolve(%__MODULE__{resolver: resolver}, ctx) do
    resolver.(ctx)
    |> unwrap()
    |> do_resolve(ctx)
  end

  defp do_resolve(entity, _ctx), do: entity

  defp unwrap({:ok, value}), do: value

  defp unwrap({:error, reason}) do
    message = if is_binary(reason), do: reason, else: inspect(reason)
    raise ArgumentError, message
  end

  defp unwrap(value), do: value
end
