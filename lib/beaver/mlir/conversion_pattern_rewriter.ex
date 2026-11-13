alias Beaver.MLIR.{CAPI, Operation, RewriterBase, Value}
alias Beaver.Native

defmodule Beaver.MLIR.ConversionPatternRewriter do
  @moduledoc "A conversion-aware pattern rewriter, including 1:N operation replacement."
  use Kinda.ResourceKind, raw_module: CAPI.Raw, codec: Native
  alias Beaver.MLIR
  defdelegate as_pattern_rewriter(rewriter), to: CAPI, as: :mlirConversionPatternRewriterAsPatternRewriter
  def as_base(%__MODULE__{} = rewriter), do: rewriter |> as_pattern_rewriter() |> MLIR.PatternRewriter.as_base()

  defdelegate convert_region_types(rewriter, region, converter),
    to: CAPI,
    as: :mlirConversionPatternRewriterConvertRegionTypes

  @doc """
  Replaces an operation through the conversion rewriter.

  Unlike `RewriterBase.replace_op/3`, the replacement types may differ from the original result types because the
  pattern's type converter defines the legal mapping.
  """
  def replace_op(%__MODULE__{} = rewriter, %Operation{} = op, replacement) do
    values =
      case replacement do
        %Operation{} = replacement_op -> replacement_op |> Operation.results() |> Enum.to_list()
        %Value{} = value -> [value]
        values when is_list(values) -> values
        other -> raise ArgumentError, "unsupported operation replacement: #{inspect(other)}"
      end

    unless Enum.all?(values, &match?(%Value{}, &1)),
      do: raise(ArgumentError, "operation replacements must be MLIR values")

    RewriterBase.replace_op_with_values(as_base(rewriter), op, length(values), Native.array(values, Value))
  end

  @spec replace_op_with_multiple(t(), Operation.t(), [[Value.t()]]) :: :ok
  def replace_op_with_multiple(%__MODULE__{} = rewriter, %Operation{} = op, ranges) when is_list(ranges) do
    unless Enum.all?(ranges, fn range -> is_list(range) and Enum.all?(range, &match?(%Value{}, &1)) end),
      do: raise(ArgumentError, "1:N replacements must be lists of MLIR values")

    result_count = op |> Operation.results() |> Enum.count()

    if length(ranges) != result_count do
      raise ArgumentError,
            "expected one replacement range per operation result (#{result_count}), got #{length(ranges)}"
    end

    sizes = Enum.map(ranges, &length/1)
    values = List.flatten(ranges)

    CAPI.mlirConversionPatternRewriterReplaceOpWithMultiple(
      rewriter,
      op,
      length(ranges),
      Native.array(sizes, Native.ISize, mut: true),
      Native.array(values, Value, mut: true)
    )
  end

  for {helper_name, arity} <- RewriterBase.helpers(),
      helper_name not in [:replace_op, "replace_op"] do
    [_ | args] = Macro.generate_arguments(arity, __MODULE__)

    if helper_name in RewriterBase.polymorphic_helper_names() do
      def unquote(:"#{helper_name}")(%__MODULE__{} = rewriter, unquote_splicing(args)),
        do: RewriterBase.unquote(:"#{helper_name}")(rewriter, unquote_splicing(args))
    else
      def unquote(:"#{helper_name}")(%__MODULE__{} = rewriter, unquote_splicing(args)),
        do: RewriterBase.unquote(:"#{helper_name}")(as_base(rewriter), unquote_splicing(args))
    end
  end
end
