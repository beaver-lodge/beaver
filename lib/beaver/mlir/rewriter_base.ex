defmodule Beaver.MLIR.RewriterBase do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  use Kinda.ResourceKind, forward_module: Beaver.Native
  alias Beaver.MLIR

  for {f, "mlirRewriterBase" <> suffix, arity} <-
        Beaver.MLIR.CAPI.__info__(:functions)
        |> Enum.map(fn {f, a} -> {f, Atom.to_string(f), a} end) do
    suffix = String.replace_prefix(suffix, "Get", "")
    helper_name = Macro.underscore(suffix)
    args = Macro.generate_arguments(arity, __MODULE__)
    defdelegate unquote(:"#{helper_name}")(unquote_splicing(args)), to: MLIR.CAPI, as: f
  end

  @doc """
  Syntactic sugar for `replace_all_uses_with/3` and `replace_all_op_uses_with_operation/3`.
  """
  def replace(%__MODULE__{} = rewriter, %MLIR.Value{} = from, %MLIR.Value{} = to) do
    replace_all_uses_with(rewriter, from, to)
  end

  def replace(%__MODULE__{} = rewriter, %MLIR.Operation{} = from, %MLIR.Operation{} = to) do
    replace_all_op_uses_with_operation(rewriter, from, to)
  end
end
