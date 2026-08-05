defmodule Beaver.Native.OpaquePtr do
  @moduledoc """
  This module defines functions working with opaque pointer, usually a `void*` in C.
  """
  import Beaver.MLIR.CAPI

  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Beaver.Native

  @doc """
  read the N bytes starting from the pointer and returns an Erlang binary
  """
  def to_binary(%__MODULE__{ref: ref}, len) do
    beaver_raw_read_opaque_ptr(ref, len)
  end

  @doc """
  read the N bytes starting from the pointer and return a resource
  """
  def to_resource(mod, %__MODULE__{ref: ptr_ref}, offset \\ 0) do
    {ref, size} =
      apply(Beaver.MLIR.CAPI.Raw, Module.concat(mod, :make_from_opaque_ptr), [ptr_ref, offset])
      |> Beaver.Native.normalize()

    {ref, size}
  end

  def null() do
    %__MODULE__{ref: beaver_raw_get_null_ptr()}
  end

  def deallocate(%__MODULE__{ref: ref}) do
    beaver_raw_deallocate_opaque_ptr(ref)
  end
end
