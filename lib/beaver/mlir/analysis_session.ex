defmodule Beaver.MLIR.AnalysisSession do
  @moduledoc """
  A process-owned cache of MLIR dominance, post-dominance, liveness and data
  layout analyses. `query/2` submits up to 256 queries in one native call.

  The caller owns the root IR and context; both must outlive the session.
  Use `mutate/2` for edits (including passes, rewriting and direct CAPI calls),
  or call `invalidate/2` after external edits. Queries and edits must not race.
  `with/2` closes the cache even if its callback raises. Query results are
  snapshots, not live analysis handles.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.{CAPI, Operation}
  import Bitwise
  @analyses [dominance: 1, post_dominance: 2, liveness: 4, data_layout: 8]
  @enforce_keys [:ref, :root, :context, :owner]
  defstruct [:ref, :root, :context, :owner]
  @type t :: %__MODULE__{}

  def create(ir) do
    root = Operation.from_module(ir)
    %__MODULE__{ref: CAPI.beaver_raw_analysis_create(root.ref), root: root, context: MLIR.context(root), owner: self()}
  end

  def with(ir, fun) when is_function(fun, 1),
    do: Beaver.Native.with_resource(create(ir), &close/1, fun)

  def close(%__MODULE__{} = session) do
    owner!(session)
    CAPI.beaver_raw_analysis_close(session.ref)
  end

  @doc "Invalidates everything except explicitly preserved analyses. Preservation is the caller's correctness contract."
  def invalidate(%__MODULE__{} = session, preserved \\ []) do
    owner!(session)
    mask = Enum.reduce(preserved, 0, fn kind, mask -> mask ||| Keyword.fetch!(@analyses, kind) end)
    CAPI.beaver_raw_analysis_invalidate(session.ref, mask)
  end

  @doc "Runs a mutation in the calling process, invalidating before and after it, including failures. Queries are rejected until the outermost mutation finishes."
  def mutate(%__MODULE__{} = session, fun) when is_function(fun, 0) do
    owner!(session)
    CAPI.beaver_raw_analysis_mutation(session.ref, true)

    try do
      fun.()
    after
      CAPI.beaver_raw_analysis_mutation(session.ref, false)
    end
  end

  @doc false
  def track(nil, fun), do: fun.()
  def track(%__MODULE__{} = session, fun), do: mutate(session, fun)

  @doc "Returns generation, cached analysis names and results for the requested queries."
  def query(%__MODULE__{} = session, queries) when is_list(queries) and length(queries) <= 256 do
    owner!(session)
    commands = Enum.map(queries, &command/1)
    {generation, cached, results} = Beaver.Execution.run(session.context, :analysis, {session.ref, commands})

    results =
      Enum.zip_with(queries, results, fn
        {:data_layout, _, _}, result -> MLIR.DataLayout.decode(result)
        _, result -> result
      end)

    %{generation: generation, cached: for({name, bit} <- @analyses, (cached &&& bit) != 0, do: name), results: results}
  end

  def query(_session, _queries), do: raise(ArgumentError, "analysis batch must contain at most 256 queries")
  def stats(session), do: session |> query([]) |> Map.delete(:results)
  def dominates?(session, lhs, rhs), do: one(session, {:dominates, lhs, rhs})
  def post_dominates?(session, lhs, rhs), do: one(session, {:post_dominates, lhs, rhs})
  def dead_after?(session, value, operation), do: one(session, {:dead_after, operation, value})
  @doc false
  def one(session, query), do: session |> query([query]) |> Map.fetch!(:results) |> hd()
  defp command({:dominates, %Operation{ref: a}, %Operation{ref: b}}), do: {0, a, b}
  defp command({:post_dominates, %Operation{ref: a}, %Operation{ref: b}}), do: {1, a, b}
  defp command({:dead_after, %Operation{ref: a}, %MLIR.Value{ref: b}}), do: {2, a, b}
  defp command({:data_layout, %Operation{ref: a}, %MLIR.Type{ref: b}}), do: {3, a, b}
  defp command(query), do: raise(ArgumentError, "unsupported analysis query: #{inspect(query)}")
  defp owner!(%{owner: owner}) when owner == self(), do: :ok
  defp owner!(_), do: raise(ArgumentError, "analysis session belongs to another process")
end
