defmodule Beaver.Native.OpaquePtr do
  alias Beaver.MLIR.CAPI

  use Fizz.ResourceKind,
    forward_module: Beaver.Native

  @doc """
  read the N bytes starting from the pointer and returns an Erlang binary
  """
  def to_binary(%__MODULE__{ref: ref}, len) do
    CAPI.beaver_raw_read_opaque_ptr(ref, len) |> Beaver.Native.check!()
  end

  @doc """
  read the N bytes starting from the pointer and return a resource
  """
  def to_resource(mod, %Beaver.Native.OpaquePtr{ref: ptr_ref}, offset \\ 0) do
    {ref, size} = Beaver.Native.forward(mod, "make_from_opaque_ptr", [ptr_ref, offset])
    Beaver.Native.forward(Beaver.Native.F32, "memref_dump", [ref])
    {ref, size}
  end
end
