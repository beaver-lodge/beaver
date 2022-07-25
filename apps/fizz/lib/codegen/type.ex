defmodule Fizz.CodeGen.Type do
  defstruct zig_t: nil, module_name: nil, kind_name: nil, delegates: [], backer: nil, fields: []

  def array_type_name(type) do
    "[*c]const " <> type
  end

  def ptr_type_name(type) do
    "[*c]" <> type
  end

  def default(root_module, type) when is_binary(type) do
    {:ok,
     %__MODULE__{zig_t: type, module_name: Module.concat(root_module, Fizz.module_name(type))}}
  end

  def gen_resource_kind(%__MODULE__{module_name: module_name, zig_t: zig_t, kind_name: kind_name}) do
    """
    pub const #{kind_name} = beam.ResourceKind(#{zig_t}, "#{module_name}");
    """
  end
end
