defmodule Beaver.MLIR.PatternRewriter do
  @moduledoc "This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}."
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native
  alias Beaver.MLIR

  defdelegate as_base(pattern), to: MLIR.CAPI, as: :mlirPatternRewriterAsBase

  for {helper_name, arity} <- Beaver.MLIR.RewriterBase.helpers() do
    [_ | args] = Macro.generate_arguments(arity, __MODULE__)

    @doc "Delegates to `Beaver.MLIR.RewriterBase.#{helper_name}/#{arity}`."
    if helper_name in Beaver.MLIR.RewriterBase.polymorphic_helper_names() do
      def unquote(:"#{helper_name}")(%__MODULE__{} = rewriter, unquote_splicing(args)) do
        MLIR.RewriterBase.unquote(:"#{helper_name}")(rewriter, unquote_splicing(args))
      end
    else
      def unquote(:"#{helper_name}")(%__MODULE__{} = rewriter, unquote_splicing(args)) do
        base = rewriter |> as_base()
        MLIR.RewriterBase.unquote(:"#{helper_name}")(base, unquote_splicing(args))
      end
    end
  end
end
