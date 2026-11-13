alias Beaver.MLIR.{Block, Operation, Value}
alias Beaver.Native

for {module, prefix, verb} <- [
      {Beaver.MLIR.DominanceInfo, "DominanceInfo", "dominates"},
      {Beaver.MLIR.PostDominanceInfo, "PostDominanceInfo", "post_dominates"}
    ] do
  defmodule module do
    @moduledoc """
    High-level access to MLIR #{if verb == "dominates", do: "dominance", else: "post-dominance"} analysis. The owned
    analysis must not outlive its IR or context. Use `with_info/2` for scoped access or call `destroy/1` explicitly.
    After mutating the analyzed IR, call `invalidate/1` before issuing another query.
    """
    alias Beaver.MLIR
    alias Beaver.MLIR.CAPI
    use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Native
    @type analyzed_ir() :: MLIR.Module.t() | Operation.t()
    @doc "Creates an owned analysis for an operation or module."
    @spec create(analyzed_ir()) :: t()
    def create(ir), do: ir |> Operation.from_module() |> CAPI.unquote(:"mlir#{prefix}Create")()
    @doc "Destroys an owned analysis."
    @spec destroy(t()) :: :ok
    defdelegate destroy(info), to: CAPI, as: :"mlir#{prefix}Destroy"
    @doc "Runs `fun` with a new analysis and always destroys it."
    @spec with_info(analyzed_ir(), (t() -> result)) :: result when result: var
    def with_info(ir, fun) when is_function(fun, 1), do: Native.with_resource(create(ir), &destroy/1, fun)

    for qualifier <- ["", "properly_"] do
      query = :"#{qualifier}#{verb}?"
      native = (qualifier <> verb) |> Macro.camelize()
      @doc "Returns whether `a` #{String.replace(qualifier <> verb, "_", " ")} `b`. Proper dominance excludes equality."
      @spec unquote(query)(t(), Operation.t(), Operation.t()) :: boolean()
      @spec unquote(query)(t(), Block.t(), Block.t()) :: boolean()
      for kind <- [Operation, Block] do
        function = :"mlir#{prefix}#{native}#{Module.split(kind) |> List.last()}"

        def unquote(query)(%__MODULE__{} = info, %unquote(kind){} = a, %unquote(kind){} = b),
          do: CAPI.unquote(function)(info, a, b) |> Native.to_term()
      end
    end

    @doc "Invalidates cached analysis information and returns `info`."
    @spec invalidate(t()) :: t()
    def invalidate(%__MODULE__{} = info) do
      :ok = CAPI.unquote(:"mlir#{prefix}Invalidate")(info)
      info
    end

    if module == Beaver.MLIR.DominanceInfo do
      @doc "Returns whether a value dominates an operation."
      @spec value_dominates?(t(), Value.t(), Operation.t()) :: boolean()
      def value_dominates?(%__MODULE__{} = info, %Value{} = value, %Operation{} = op),
        do: CAPI.mlirDominanceInfoValueDominates(info, value, op) |> Native.to_term()

      @doc "Returns whether a value properly dominates an operation."
      @spec value_properly_dominates?(t(), Value.t(), Operation.t()) :: boolean()
      def value_properly_dominates?(%__MODULE__{} = info, %Value{} = value, %Operation{} = op),
        do: CAPI.mlirDominanceInfoValueProperlyDominates(info, value, op) |> Native.to_term()

      @doc "Returns the nearest common dominator, or `nil` when none exists."
      @spec nearest_common_dominator(t(), Block.t(), Block.t()) :: Block.t() | nil
      def nearest_common_dominator(%__MODULE__{} = info, %Block{} = a, %Block{} = b) do
        block = CAPI.mlirDominanceInfoFindNearestCommonDominator(info, a, b)
        if MLIR.null?(block), do: nil, else: block
      end

      @doc "Returns whether a block is reachable from the entry block of its region."
      @spec reachable_from_entry?(t(), Block.t()) :: boolean()
      def reachable_from_entry?(%__MODULE__{} = info, %Block{} = block),
        do: CAPI.mlirDominanceInfoIsReachableFromEntry(info, block) |> Native.to_term()
    end
  end
end
