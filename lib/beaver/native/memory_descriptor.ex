defmodule Beaver.Native.Memory.Descriptor do
  alias Beaver.Native
  @moduledoc false
  @type t() :: %__MODULE__{ref: reference(), descriptor_kind: atom()}
  defstruct [:ref, :descriptor_kind]

  def make(kind, {_allocated, _aligned, _offset, _sizes, _strides} = args)
      when is_atom(kind) do
    %__MODULE__{
      ref:
        Native.forward(
          kind,
          "make",
          Tuple.to_list(args)
        ),
      descriptor_kind: kind
    }
  end

  def aligned(%__MODULE__{ref: ref, descriptor_kind: descriptor_kind}) do
    struct!(Native.OpaquePtr,
      ref: Native.forward(descriptor_kind, :aligned, [ref])
    )
  end

  def allocated(%__MODULE__{ref: ref, descriptor_kind: descriptor_kind}) do
    struct!(Native.OpaquePtr,
      ref: Native.forward(descriptor_kind, :allocated, [ref])
    )
  end

  def opaque_ptr(%__MODULE__{ref: ref, descriptor_kind: descriptor_kind}) do
    struct!(Native.OpaquePtr,
      ref: Native.forward(descriptor_kind, :opaque_ptr, [ref])
    )
  end
end
