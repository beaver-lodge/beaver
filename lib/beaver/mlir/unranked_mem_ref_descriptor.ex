defmodule Beaver.MLIR.UnrankedMemRefDescriptor do
  @moduledoc """
  This module defines functions working with MLIR UnrankedMemRefDescriptor runtime ABI.

  UnrankedMemRefDescriptor represents an unranked memref descriptor that can hold
  memory reference information for dynamically shaped arrays. It provides access
  to the rank, sizes, strides, and offset of the memory reference. Due to the lack
  of a ctypes/libffi equivalent in Elixir, in Beaver we use unranked memref descriptors
  primarily as ABI.
  """
  alias Beaver.MLIR
  use Kinda.ResourceKind, forward_module: Beaver.Native

  @doc """
  Creates an empty unranked memref descriptor with the specified rank.
  """
  def empty(rank) when is_integer(rank) and rank >= 0 do
    %__MODULE__{ref: MLIR.CAPI.beaver_raw_unranked_memref_descriptor_empty(rank)}
  end

  @doc """
  Gets the rank of the unranked memref descriptor.
  """
  def rank(%__MODULE__{ref: descriptor}) do
    MLIR.CAPI.beaver_raw_unranked_memref_descriptor_get_rank(descriptor)
  end

  @doc """
  Gets the offset of the unranked memref descriptor.
  """
  def offset(%__MODULE__{ref: descriptor}) do
    MLIR.CAPI.beaver_raw_unranked_memref_descriptor_get_offset(descriptor)
  end

  @doc """
  Gets the sizes of the unranked memref descriptor as a list.

  For rank 0 descriptors, returns an empty list.
  """
  def sizes(%__MODULE__{ref: descriptor}) do
    MLIR.CAPI.beaver_raw_unranked_memref_descriptor_get_sizes(descriptor)
  end

  @doc """
  Gets the strides of the unranked memref descriptor as a list.

  For rank 0 descriptors, returns an empty list.
  """
  def strides(%__MODULE__{ref: descriptor}) do
    MLIR.CAPI.beaver_raw_unranked_memref_descriptor_get_strides(descriptor)
  end

  @doc """
  Frees the memory allocated for the unranked memref descriptor if it was
  allocated by the runtime.
  This is useful when the memory was allocated by a JIT-compiled function
  and needs to be freed after use.
  """
  def free(%__MODULE__{ref: descriptor}) do
    MLIR.CAPI.beaver_raw_unranked_memref_descriptor_free_allocated(descriptor)
  end

  @doc false
  def opaque_ptr(%__MODULE__{ref: descriptor}) do
    ref = Beaver.Native.forward(__MODULE__, :opaque_ptr, [descriptor])
    struct!(Beaver.Native.OpaquePtr, ref: ref)
  end
end
