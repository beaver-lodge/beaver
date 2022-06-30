defmodule Beaver.MLIR.ODS do
  @moduledoc """
  ODS helper functions to work with special concepts in ODS.
  """

  alias Beaver.MLIR
  alias MLIR.{Type, Attribute}

  @doc """
  Generate attribute for operand_segment_sizes
  """
  def operand_segment_sizes(sizes) when is_list(sizes) do
    sizes = Enum.map(sizes, fn s -> Attribute.integer(Type.i(32), s) end)

    Attribute.dense_elements(
      sizes,
      Type.vector([length(sizes)], Type.i(32))
    )
  end

  defdelegate result_segment_sizes(sizes), to: __MODULE__, as: :operand_segment_sizes
end
