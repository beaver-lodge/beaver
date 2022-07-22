defmodule Fizz.CodeGen.Type do
  defstruct zig_t: nil, module_name: nil, delegates: [], backer: nil, fields: []

  def array_type_name(type) do
    "[*c]const " <> type
  end

  def ptr_type_name(type) do
    "[*c]" <> type
  end

  def default(type) do
    {:ok, %__MODULE__{zig_t: type, module_name: Fizz.module_name(type)}}
  end
end
