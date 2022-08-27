defmodule Beaver.Native.Memory do
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

  defp extract_mod_from_opts(opts) do
    mod = Keyword.fetch!(opts, :type)

    case mod do
      {:u, 8} ->
        Beaver.Native.U8

      {:u, 16} ->
        Beaver.Native.U16

      {:u, 32} ->
        Beaver.Native.U32

      {:u, 64} ->
        Beaver.Native.U64

      {:s, 8} ->
        Beaver.Native.I8

      {:s, 16} ->
        Beaver.Native.I16

      {:s, 32} ->
        Beaver.Native.I32

      {:s, 64} ->
        Beaver.Native.I64

      {:f, 32} ->
        Beaver.Native.F32

      {:c, 64} ->
        Beaver.Native.Complex.F32

      mod when is_atom(mod) ->
        mod
    end
  end

  def new(data, opts \\ [offset: 0])

  def new(data, opts) when is_list(data) or is_binary(data) do
    mod = extract_mod_from_opts(opts)

    array =
      if data do
        mod.array(data, mut: true)
      end

    new(array, opts)
  end

  def new(%Beaver.Native.Array{ref: ref, element_kind: mod} = array, opts) do
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
        __MODULE__.Descriptor.make(shape_to_descriptor_kind(mod, sizes), [
          ref,
          ref,
          offset,
          sizes,
          strides
        ])
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
        __MODULE__.Descriptor.make(shape_to_descriptor_kind(mod, sizes), [
          nil,
          nil,
          offset,
          sizes,
          strides
        ])
    }
  end

  @doc """
  return a opaque pointer to the memory
  """
  def aligned(%__MODULE__{
        descriptor: %__MODULE__.Descriptor{ref: descriptor_ref, kind: descriptor_kind},
        storage: storage
      }) do
    ptr =
      struct!(Beaver.Native.OpaquePtr,
        ref: Beaver.Native.forward(descriptor_kind, :aligned, [descriptor_ref])
      )

    if storage do
      ptr |> Beaver.Native.bag(storage)
    else
      ptr
    end
  end

  def allocated(%__MODULE__{
        descriptor: %__MODULE__.Descriptor{ref: descriptor_ref, kind: descriptor_kind},
        storage: storage
      }) do
    ptr =
      struct!(Beaver.Native.OpaquePtr,
        ref: Beaver.Native.forward(descriptor_kind, :allocated, [descriptor_ref])
      )

    if storage do
      ptr |> Beaver.Native.bag(storage)
    else
      ptr
    end
  end

  @doc """
  return a opaque pointer to the memory descriptor. Usually used in the invoking of a generated function.
  If it is a array, will return the pointer of the array to mimic a struct of packed memory descriptors
  """
  def descriptor_ptr(%__MODULE__{
        descriptor: %__MODULE__.Descriptor{ref: descriptor_ref, kind: descriptor_kind},
        storage: storage
      }) do
    struct!(Beaver.Native.OpaquePtr,
      ref: Beaver.Native.forward(descriptor_kind, :opaque_ptr, [descriptor_ref])
    )
    |> Beaver.Native.bag(storage)
  end

  def descriptor_ptr(%Beaver.Native.Array{ref: ref, element_kind: element_kind} = array) do
    ref = Beaver.Native.forward(element_kind, :ptr_to_opaque, [ref])
    struct!(Beaver.Native.OpaquePtr, ref: ref) |> Beaver.Native.bag(array)
  end

  @doc """
  take ownership of the memory the descriptor points to
  """
  def own_allocated(
        %__MODULE__{
          storage: nil
        } = m
      ) do
    owner =
      m
      |> allocated
      |> Beaver.Native.PtrOwner.new()

    %{m | storage: owner}
  end
end
