defmodule Beaver.Native.Memory do
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

  def new(data, opts \\ [offset: 0])

  def new(data, opts) when is_list(data) or is_binary(data) do
    mod = Keyword.fetch!(opts, :type)

    mod =
      case mod do
        {:s, 64} ->
          Beaver.Native.I64

        _ ->
          mod
      end

    # TODO: if no sizes given, create a unranked memref
    offset = Keyword.get(opts, :offset, 0)
    sizes = Keyword.fetch!(opts, :sizes)

    strides =
      Keyword.get(
        opts,
        :strides,
        dense_strides(sizes)
      )

    array = %{ref: ref} = mod.array(data, mut: true)

    %__MODULE__{
      storage: array,
      descriptor_ref:
        Beaver.Native.forward(mod, "memref_create", [ref, ref, offset, sizes, strides])
    }
  end

  def new(%Beaver.Native.Array{ref: ref, element_module: mod} = array, opts) do
    offset = Keyword.get(opts, :offset, 0)
    sizes = Keyword.fetch!(opts, :sizes)

    strides =
      Keyword.get(
        opts,
        :strides,
        dense_strides(sizes)
      )

    %__MODULE__{
      storage: array,
      descriptor_ref:
        Beaver.Native.forward(mod, "memref_create", [ref, ref, offset, sizes, strides])
    }
  end

  @doc """
  return a opaque pointer to the memory
  """
  def aligned(%__MODULE__{
        descriptor_ref: descriptor_ref,
        storage: %{element_module: element_module} = array
      }) do
    struct!(Beaver.Native.OpaquePtr,
      ref: Beaver.Native.forward(element_module, "memref_aligned", [descriptor_ref])
    )
    |> Beaver.Native.bag(array)
  end

  @doc """
  return a opaque pointer to the memory descriptor. Usually used in the invoking of a generated function.
  """
  def descriptor_ptr(%__MODULE__{
        descriptor_ref: descriptor_ref,
        storage: %{element_module: element_module} = array
      }) do
    struct!(Beaver.Native.OpaquePtr,
      ref: Beaver.Native.forward(element_module, "memref_opaque_ptr", [descriptor_ref])
    )
    |> Beaver.Native.bag(array)
  end
end
