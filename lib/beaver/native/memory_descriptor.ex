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

  defp call_ptr_func(%__MODULE__{ref: ref, descriptor_kind: k}, func) do
    struct!(Native.OpaquePtr,
      ref: Native.forward(k, func, [ref])
    )
  end

  def aligned(d) do
    call_ptr_func(d, :aligned)
  end

  def allocated(d) do
    call_ptr_func(d, :allocated)
  end

  def opaque_ptr(d) do
    call_ptr_func(d, :opaque_ptr)
  end

  def dump(%__MODULE__{ref: ref, descriptor_kind: k}) do
    Native.forward(k, :dump, [ref])
  end

  def offset(%__MODULE__{ref: ref, descriptor_kind: k}) do
    Native.forward(k, :offset, [ref])
  end
end
