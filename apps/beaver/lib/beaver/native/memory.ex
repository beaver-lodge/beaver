defmodule Beaver.Native.Memory do
  alias Beaver.MLIR.CAPI

  @moduledoc """
  A piece of memory managed by BEAM and can by addressed by a generated native function as MLIR MemRef descriptor
  """

  defstruct storage: nil, descriptor_ref: nil

  defp dim_product(dims) when is_list(dims) and length(dims) > 0 do
    dims |> Enum.reduce(&*/2)
  end

  defp infer_dense_strides([_], strides) when is_list(strides) do
    strides ++ [1]
  end

  defp infer_dense_strides([_ | tail], strides) when is_list(strides) do
    infer_dense_strides(tail, strides ++ [dim_product(tail)])
  end

  def dense_strides([]) do
    []
  end

  def dense_strides(shape) when is_list(shape) do
    infer_dense_strides(shape, [])
  end

  def new(data, opts \\ [offset: 0]) when is_list(data) do
    offset = Keyword.get(opts, :offset, 0)
    mod = Keyword.fetch!(opts, :type)
    # TODO: if no sizes given, create a unranked memref
    sizes = Keyword.fetch!(opts, :sizes)

    strides =
      Keyword.get(
        opts,
        :strides,
        dense_strides(sizes)
      )

    array = mod.array(data, mut: true)

    %__MODULE__{
      storage: array,
      descriptor_ref:
        mod.memref(array.ref, array.ref, offset, sizes, strides) |> Beaver.Native.check!()
    }
  end

  def ref() do
  end

  @doc """
  return a opaque pointer to the memory
  """
  def aligned(%__MODULE__{
        descriptor_ref: descriptor_ref,
        storage: %Beaver.Native.Array{element_module: element_module}
      }) do
    apply(CAPI, Module.concat(element_module, "memref_aligned"), [descriptor_ref])
    |> Beaver.Native.check!()
  end
end
