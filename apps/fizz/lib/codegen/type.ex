defmodule Fizz.CodeGen.Type do
  defstruct zig_t: nil, module_name: nil, delegates: [], backer: nil, fields: []

  def array_type_name(type) do
    "[*c]const " <> type
  end

  def ptr_type_name(type) do
    "[*c]" <> type
  end

  def default(type) when is_binary(type) do
    {:ok, %__MODULE__{zig_t: type, module_name: Fizz.module_name(type)}}
  end

  def gen_resource_struct(%__MODULE__{module_name: module_name, zig_t: zig_t}) do
    """
    pub const #{module_name} = beam.get_element_struct(#{zig_t}, root_module ++ ".#{module_name}");
    """
  end
end
