defmodule Beaver.MLIR.OptimizationRemarks do
  @moduledoc """
  Optimization explanations emitted by MLIR passes. `attach/2` sends copied JSON
  messages to the calling process, tagged with the returned reference. `drain/1`
  decodes pending messages; finalize the engine to flush `policy: :final` remarks.
  The context owns the callback and releases it on finalization or destruction.
  Attach/finalize only while no other thread is using the context.
  """
  alias Beaver.MLIR.{CAPI, Context, Location, StringRef}
  alias Beaver.Native
  @kinds %{passed: 1, missed: 2, failure: 3, analysis: 4}

  def enabled?(%Context{} = ctx), do: CAPI.beaverContextHasOptimizationRemarks(ctx) |> Native.to_term()

  @doc "Enables remarks; filters use LLVM regular expressions on category names."
  def attach(%Context{} = ctx, opts \\ []) do
    opts = Keyword.validate!(opts, policy: :all, filters: [all: ".*"])
    policy = Keyword.fetch!(opts, :policy)
    filters = Keyword.fetch!(opts, :filters)
    unless policy in [:all, :final], do: raise(ArgumentError, "expected remark policy :all or :final")

    unless Keyword.keyword?(filters) and
             Enum.all?(filters, fn {key, value} ->
               key in [:all, :passed, :missed, :analysis, :failed] and is_binary(value)
             end),
           do: raise(ArgumentError, "invalid optimization remark category filters")

    ref = make_ref()

    if CAPI.beaver_raw_optimization_remarks_attach(ctx.ref, ref, JSON.encode!(Map.new(filters)), policy == :final),
      do: ref,
      else: raise(ArgumentError, "cannot enable remarks: unsupported LLVM, invalid filter, or engine already enabled")
  end

  @doc "Finalizes the engine, reporting postponed remarks and releasing the callback."
  def finalize(%Context{} = ctx), do: CAPI.beaverContextFinalizeOptimizationRemarks(ctx)

  @doc "Returns already-delivered remarks for this reference in emission order."
  def drain(ref) when is_reference(ref), do: drain(ref, [])

  defp drain(ref, entries) do
    receive do
      {:mlir_optimization_remark, ^ref, json} -> drain(ref, [JSON.decode!(json) | entries])
    after
      0 -> Enum.reverse(entries)
    end
  end

  @doc "Emits a remark from an Elixir pass, returning false if the category is filtered out."
  def emit(%Location{} = location, kind, name, opts \\ []) when is_map_key(@kinds, kind) and is_binary(name) do
    opts = Keyword.validate!(opts, category: "beaver", subcategory: "", function: "", message: "", args: [])

    unless Enum.all?(Keyword.drop(opts, [:args]), fn {_, value} -> is_binary(value) end),
      do: raise(ArgumentError, "remark category, subcategory, function and message must be strings")

    args = Enum.map(Keyword.fetch!(opts, :args), fn {key, value} -> [to_string(key), to_string(value)] end)
    payload = opts |> Map.new() |> Map.merge(%{kind: Map.fetch!(@kinds, kind), name: name, args: args})
    CAPI.beaverEmitOptimizationRemark(location, StringRef.create(JSON.encode!(payload))) |> Native.to_term()
  end
end
