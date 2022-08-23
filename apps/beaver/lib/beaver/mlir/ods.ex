defmodule Beaver.MLIR.ODS do
  @moduledoc """
  ODS helper functions to work with special concepts in ODS.
  """

  alias Beaver.MLIR
  alias MLIR.{Attribute}

  @doc """
  Generate attribute for operand_segment_sizes
  """
  def operand_segment_sizes(sizes) when is_list(sizes) do
    Attribute.dense_array(sizes)
  end

  defdelegate result_segment_sizes(sizes), to: __MODULE__, as: :operand_segment_sizes
end
