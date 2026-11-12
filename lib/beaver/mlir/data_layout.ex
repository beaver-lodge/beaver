defmodule Beaver.MLIR.DataLayout do
  @moduledoc """
  Queries the closest MLIR data layout at an operation, using an AnalysisSession cache. Sizes retain both their
  known minimum and whether they are scalable. Layout changes, including changes to enclosing operations, require
  session invalidation. Index and pointer widths come from MLIR, never the host VM.
  """
  alias Beaver.MLIR.{AnalysisSession, Operation}

  def type_layout(%AnalysisSession{} = session, type, opts \\ []) do
    scope = opts |> Keyword.get(:at, session.root) |> Operation.from_module()
    AnalysisSession.one(session, {:data_layout, scope, type})
  end

  @doc false
  def decode({size, scalable, bits, bits_scalable, alloc, alloc_scalable, abi, preferred, index}) do
    %{
      size: size(size, scalable),
      size_in_bits: size(bits, bits_scalable),
      allocation_size: size(alloc, alloc_scalable),
      abi_alignment: abi,
      preferred_alignment: preferred,
      index_bitwidth: if(index == 0, do: nil, else: index)
    }
  end

  defp size(minimum, scalable), do: %{minimum: minimum, scalable?: scalable != 0}
end
