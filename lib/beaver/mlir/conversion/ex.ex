defmodule Beaver.MLIR.Conversion.Ex do
  @moduledoc """
  Reference Ex lowering. `plan(layers: ...)` preserves unselected owners;
  `finish!/1` completes lowering and rejects unresolved type bridges.
  See [layer contracts](dialect-conversion.html#ex-provider-boundary).
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.{CAPI, Operation}
  alias Beaver.MLIR.Conversion.Ex.Values
  alias Beaver.MLIR.Conversion.Plan
  alias Beaver.MLIR.Dialect.Ex, as: ExDialect
  alias Beaver.Walker

  @term_types ~w(!ex.term !ex.bound !ex.unbound)

  @doc "Completes remaining Ex lowering and reconciles the type bridges left by layer plans."
  def finish!(ir, opts \\ []) do
    ir =
      ir
      |> then(&Plan.run!(plan(opts), &1))
      |> Beaver.Composer.append("reconcile-unrealized-casts")
      |> Beaver.Composer.run!()

    Walker.prewalk(ir, fn
      %Operation{} = op ->
        if Operation.name(op) == "builtin.unrealized_conversion_cast",
          do: raise(ArgumentError, "unresolved Ex type bridge")

        op

      element ->
        element
    end)

    ir
  end

  @doc "Returns a reference conversion plan for all Ex owners, or only the selected `:layers`."
  @spec plan(keyword()) :: Plan.t()
  def plan(opts \\ []) do
    {layers, opts} = Keyword.pop(opts, :layers, ExDialect.layers())
    {runtime_namespace, opts} = Keyword.pop(opts, :runtime_namespace)
    layers = List.wrap(layers)

    unless layers != [] and layers -- ExDialect.layers() == [],
      do: raise(ArgumentError, "invalid Ex conversion layers: #{inspect(layers)}")

    selected = Enum.flat_map(layers, &ExDialect.operations/1) |> MapSet.new()

    defaults = [mode: :full, folding_mode: :after_patterns, build_materializations: length(layers) == 3]

    Plan.new(Keyword.merge(defaults, opts))
    |> Plan.add_legal_dialect("builtin")
    |> Plan.add_legal_dialect("func")
    |> Plan.add_legal_dialect("arith")
    |> Plan.add_legal_dialect("cf")
    |> Plan.add_legal_dialect("scf")
    |> Plan.add_legal_dialect("llvm")
    |> Plan.add_illegal_dialect("ex")
    |> retain_other_layers(selected)
    |> Plan.add_conversion_map(["i64"], "i64")
    |> Values.configure(runtime_namespace, selected)
    |> Plan.add_pattern_population(&populate_native_scalar_patterns/2, version: "1.0")
    |> add_conversion_patterns(selected, runtime_namespace)
  end

  defp retain_other_layers(plan, selected),
    do: Enum.reduce(ExDialect.operations() -- MapSet.to_list(selected), plan, &Plan.add_legal_op(&2, &1))

  defp add_conversion_patterns(plan, selected, namespace) do
    Enum.reduce(Enum.sort(selected), plan, fn root, plan ->
      if root in ~w(ex.call ex.return ex.var ex.func) or String.starts_with?(root, "ex.btt_"),
        do:
          Plan.add_conversion_pattern(plan, root, &Values.lower(&1, &2, &3, nil, namespace),
            version: "3.0/#{namespace}"
          ),
        else: plan
    end)
  end

  defp populate_native_scalar_patterns(patterns, converter),
    do: CAPI.beaverPopulateExScalarConversionPatterns(patterns.ref, converter.ref)

  @doc "Preserves scalar types; abstract language terms require a production provider."
  def convert_type(type) do
    if MLIR.to_string(type) in @term_types,
      do: raise(ArgumentError, "abstract Ex terms require a runtime provider; the tagged-word ABI was retired")

    type
  end
end
