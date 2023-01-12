defmodule Beaver.Native.Memory.Descriptor do
  @moduledoc false
  @type t() :: %__MODULE__{ref: reference(), descriptor_kind: atom()}
  defstruct [:ref, :descriptor_kind]

  def make(kind, {_allocated, _aligned, _offset, _sizes, _strides} = args)
      when is_atom(kind) do
    %__MODULE__{
      ref:
        Beaver.Native.forward(
          kind,
          "make",
          Tuple.to_list(args)
        ),
      descriptor_kind: kind
    }
  end
end
