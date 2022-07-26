defmodule Fizz.CodeGen.Resource do
  alias Fizz.CodeGen.Type

  def resource_type_struct("[*c]const " <> type, %{} = resource_kind_map) do
    mod = Map.fetch!(resource_kind_map, type)
    "#{mod}.Array"
  end

  def resource_type_struct("[*c]" <> type, %{} = resource_kind_map) do
    mod = Map.fetch!(resource_kind_map, type)
    "#{mod}.Ptr"
  end

  def resource_type_struct(type, %{} = resource_kind_map) do
    mod = Map.fetch!(resource_kind_map, type)
    "#{mod}"
  end

  def resource_type_resource_kind(type, %{} = resource_kind_map) do
    resource_type_struct(type, resource_kind_map) <> ".resource"
  end

  def resource_type_var(type, %{} = resource_kind_map) do
    resource_type_resource_kind(type, resource_kind_map) <> ".t"
  end

  def resource_open(%Type{kind_name: kind_name}) do
    """
    #{kind_name}.open_all(env);
    """
  end

  defmacro gen_resource_functions(module_name) do
    for {f, a} <- [
          ptr: 1,
          opaque_ptr: 1,
          array: 1,
          mut_array: 1,
          primitive: 1,
          make: 1,
          create_memref: 5
        ] do
      quote bind_quoted: [f: f, a: a, module_name: module_name] do
        name = Module.concat(module_name, f)
        args = List.duplicate({:_, [if_undefined: :apply], Elixir}, a)
        def unquote(name)(unquote_splicing(args)), do: raise("NIF not loaded")
      end
    end
    |> List.flatten()
  end
end
