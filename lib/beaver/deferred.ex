alias Beaver.MLIR.Context

defmodule Beaver.Deferred do
  @moduledoc """
  Context-bound MLIR construction. Builders return `%Beaver.Deferred{}` without
  `:ctx`, or concrete entities with it. Unlike ordinary one-argument callbacks,
  deferred values can be materialized by `resolve/2`, which also checks the
  context ownership of concrete entities.
  """
  alias Beaver.MLIR

  @enforce_keys [:resolver]
  defstruct [:resolver]

  @type context_arg() :: Context.t()
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
  Runs `resolver` with `:ctx`, or defers it when that option is absent.
  Explicit `ctx: nil` and other non-context values are rejected.
  """
  @spec from_opts(opts(), (context_arg() -> value)) :: contextual(value) when value: var
  def from_opts(opts, resolver) when is_list(opts) and is_function(resolver, 1) do
    case context(opts) do
      nil -> defer(resolver)
      ctx -> resolver.(ctx) |> Context.ensure_same!(ctx)
    end
  end

  @doc "Returns the optional context in `opts`, validating an explicitly supplied value."
  @spec context(opts()) :: context_arg() | nil
  def context(opts) when is_list(opts) do
    case Keyword.fetch(opts, :ctx) do
      :error -> nil
      {:ok, %Context{} = ctx} -> ctx
      {:ok, invalid} -> raise ArgumentError, "expected :ctx to be an MLIR context, got: #{inspect(invalid)}"
    end
  end

  @spec fetch_insertion_point(keyword()) ::
          MLIR.Block.t() | MLIR.PatternRewriter.t() | MLIR.RewriterBase.t() | Macro.t() | nil
  def fetch_insertion_point(opts) do
    for key <- [:block, :blk] do
      if opts[key], do: raise(ArgumentError, "use :ip instead of :#{key} as key")
    end

    opts[:ip]
  end

  @doc """
  Resolves deferred values in `ctx` and checks ownership of concrete entities.
  Resolver successes `{:ok, value}` are unwrapped; `{:error, reason}` raises
  `ArgumentError`.
  """
  @spec resolve(contextual(value), context_arg()) :: value when value: var
  def resolve(resolver, %Context{}) when is_function(resolver, 1),
    do: raise(ArgumentError, "bare context resolvers are not deferred values; wrap the function with defer/1")

  def resolve(value, %Context{} = ctx), do: value |> do_resolve(ctx) |> Context.ensure_same!(ctx)

  def resolve(_value, invalid), do: raise(ArgumentError, "expected an MLIR context, got: #{inspect(invalid)}")

  defp do_resolve({:parametric, _, _, deferred}, ctx), do: do_resolve(deferred, ctx)

  defp do_resolve(%__MODULE__{resolver: resolver}, ctx), do: resolver.(ctx) |> unwrap() |> do_resolve(ctx)

  defp do_resolve(entity, _ctx), do: entity

  defp unwrap({:ok, value}), do: value

  defp unwrap({:error, reason}) do
    message = if is_binary(reason), do: reason, else: inspect(reason)
    raise ArgumentError, message
  end

  defp unwrap(value), do: value
end
