defmodule Beaver.MLIR.ConversionPatternRewriter do
  @moduledoc """
  A conversion-aware pattern rewriter, including 1:N operation replacement.
  """

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  alias Beaver.MLIR

  defdelegate as_pattern_rewriter(rewriter),
    to: MLIR.CAPI,
    as: :mlirConversionPatternRewriterAsPatternRewriter

  def as_base(%__MODULE__{} = rewriter) do
    rewriter |> as_pattern_rewriter() |> MLIR.PatternRewriter.as_base()
  end

  defdelegate convert_region_types(rewriter, region, converter),
    to: MLIR.CAPI,
    as: :mlirConversionPatternRewriterConvertRegionTypes

  @doc """
  Replaces an operation through the conversion rewriter.

  Unlike `RewriterBase.replace_op/3`, the replacement types may differ from
  the original result types because the pattern's type converter defines the
  legal mapping.
  """
  def replace_op(%__MODULE__{} = rewriter, %MLIR.Operation{} = operation, replacement) do
    values =
      case replacement do
        %MLIR.Operation{} = replacement_op ->
          replacement_op |> MLIR.Operation.results() |> Enum.to_list()

        %MLIR.Value{} = value ->
          [value]

        values when is_list(values) ->
          values

        other ->
          raise ArgumentError, "unsupported operation replacement: #{inspect(other)}"
      end

    unless Enum.all?(values, &match?(%MLIR.Value{}, &1)) do
      raise ArgumentError, "operation replacements must be MLIR values"
    end

    MLIR.RewriterBase.replace_op_with_values(
      as_base(rewriter),
      operation,
      length(values),
      Beaver.Native.array(values, MLIR.Value)
    )
  end

  @spec replace_op_with_multiple(t(), MLIR.Operation.t(), [[MLIR.Value.t()]]) :: :ok
  def replace_op_with_multiple(%__MODULE__{} = rewriter, %MLIR.Operation{} = operation, ranges)
      when is_list(ranges) do
    unless Enum.all?(ranges, fn range ->
             is_list(range) and Enum.all?(range, &match?(%MLIR.Value{}, &1))
           end) do
      raise ArgumentError, "1:N replacements must be lists of MLIR values"
    end

    result_count = operation |> MLIR.Operation.results() |> Enum.count()

    if length(ranges) != result_count do
      raise ArgumentError,
            "expected one replacement range per operation result (#{result_count}), got #{length(ranges)}"
    end

    sizes = Enum.map(ranges, &length/1)
    values = List.flatten(ranges)

    MLIR.CAPI.mlirConversionPatternRewriterReplaceOpWithMultiple(
      rewriter,
      operation,
      length(ranges),
      Beaver.Native.array(sizes, Beaver.Native.ISize, mut: true),
      Beaver.Native.array(values, MLIR.Value, mut: true)
    )
  end

  for {helper_name, arity} <- Beaver.MLIR.RewriterBase.helpers(),
      helper_name not in [:replace_op, "replace_op"] do
    [_ | args] = Macro.generate_arguments(arity, __MODULE__)

    def unquote(:"#{helper_name}")(%__MODULE__{} = rewriter, unquote_splicing(args)) do
      MLIR.RewriterBase.unquote(:"#{helper_name}")(as_base(rewriter), unquote_splicing(args))
    end
  end
end
