defmodule Beaver.Native.PtrOwner do
  alias Beaver.MLIR.CAPI

  @moduledoc """
  owning a pointer resource and delete it when BEAM collect the resource term
  """
  defstruct ref: nil, kind: Beaver.Native.PtrOwner

  def(new(%Beaver.Native.OpaquePtr{ref: ptr_ref})) do
    owner_ref =
      CAPI.beaver_raw_own_opaque_ptr(ptr_ref)
      |> Beaver.Native.check!()

    %__MODULE__{ref: owner_ref}
  end
end
