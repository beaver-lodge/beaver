defmodule Beaver.Native.Memory do
  alias Beaver.Native

  @moduledoc """
  A piece of memory managed by BEAM and can by addressed by a generated native function as MLIR MemRef descriptor
  """

  @enforce_keys [:descriptor]
  defstruct storage: nil, descriptor: nil

  defp shape_to_descriptor_kind(type, []) do
    Module.concat([type, MemRef.DescriptorUnranked])
  end

  defp shape_to_descriptor_kind(type, list) when is_list(list) do
    rank = length(list)
    Module.concat([type, MemRef, "Descriptor#{rank}D"])
  end

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

  defp pair_to_mod({:u, 8}), do: Native.U8
  defp pair_to_mod({:u, 16}), do: Native.U16
  defp pair_to_mod({:u, 32}), do: Native.U32
  defp pair_to_mod({:u, 64}), do: Native.U64
  defp pair_to_mod({:s, 8}), do: Native.I8
  defp pair_to_mod({:s, 16}), do: Native.I16
  defp pair_to_mod({:s, 32}), do: Native.I32
  defp pair_to_mod({:s, 64}), do: Native.I64
  defp pair_to_mod({:f, 32}), do: Native.F32
  defp pair_to_mod({:c, 64}), do: Native.Complex.F32
  defp pair_to_mod(mod) when is_atom(mod), do: mod

  defp extract_mod_from_opts(opts) do
    Keyword.fetch!(opts, :type)
    |> pair_to_mod()
  end

  def new(data, opts \\ [offset: 0])

  def new(data, opts) when is_list(data) or is_binary(data) do
    mod = extract_mod_from_opts(opts)

    array =
      if data do
        Native.array(data, mod, mut: true)
      end

    new(array, opts)
  end

  def new(%Native.Array{ref: ref, element_kind: mod} = array, opts) do
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
      descriptor:
        __MODULE__.Descriptor.make(shape_to_descriptor_kind(mod, sizes), {
          ref,
          ref,
          offset,
          sizes,
          strides
        })
    }
  end

  def new(nil = storage, opts) do
    offset = Keyword.get(opts, :offset, 0)
    sizes = Keyword.fetch!(opts, :sizes)
    mod = extract_mod_from_opts(opts)

    strides =
      Keyword.get(
        opts,
        :strides,
        dense_strides(sizes)
      )

    %__MODULE__{
      storage: storage,
      descriptor:
        __MODULE__.Descriptor.make(shape_to_descriptor_kind(mod, sizes), {
          nil,
          nil,
          offset,
          sizes,
          strides
        })
    }
  end

  @doc """
  return a opaque pointer to the memory
  """
  def aligned(%__MODULE__{
        descriptor: d
      }) do
    __MODULE__.Descriptor.aligned(d)
  end

  def allocated(%__MODULE__{
        descriptor: d
      }) do
    __MODULE__.Descriptor.allocated(d)
  end

  @doc """
  return a opaque pointer to the memory descriptor. Usually used in the invoking of a generated function.
  If it is a array, will return the pointer of the array to mimic a struct of packed memory descriptors
  """
  def descriptor_ptr(%__MODULE__{
        descriptor: d
      }) do
    __MODULE__.Descriptor.opaque_ptr(d)
  end

  # if this is an array, this should be packed memory descriptors for tuple
  def descriptor_ptr(%Native.Array{ref: ref, element_kind: element_kind}) do
    ref = Native.forward(element_kind, :ptr_to_opaque, [ref])
    struct!(Native.OpaquePtr, ref: ref)
  end

  def deallocate(
        %__MODULE__{
          storage: nil
        } = m
      ) do
    m |> allocated |> Beaver.Native.OpaquePtr.deallocate()
  end
end
