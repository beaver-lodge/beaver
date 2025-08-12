defmodule Beaver.MLIR.ODS do
  @moduledoc """
  ODS helper functions to work with special concepts in ODS.
  """

  alias Beaver.MLIR
  alias MLIR.{Attribute}

  @doc """
  Generate attribute for operand_segment_sizes
  """
  def operand_segment_sizes(sizes, opts \\ []) when is_list(sizes) do
    Beaver.Deferred.from_opts(
      opts,
      &Attribute.dense_array(sizes, Beaver.Native.I32, ctx: &1)
    )
  end

  defdelegate result_segment_sizes(sizes), to: __MODULE__, as: :operand_segment_sizes
end
