defmodule Fizz.CodeGen.Resource do
  alias Fizz.CodeGen.Type

  def resource_type_struct("[*c]const " <> type, %{} = resource_struct_map) do
    mod = Map.fetch!(resource_struct_map, type)
    "#{mod}.Array.resource"
  end

  def resource_type_struct("[*c]" <> type, %{} = resource_struct_map) do
    mod = Map.fetch!(resource_struct_map, type)
    "#{mod}.Ptr.resource"
  end

  def resource_type_struct(type, %{} = resource_struct_map) do
    mod = Map.fetch!(resource_struct_map, type)
    "#{mod}.resource"
  end

  def resource_type_var(type, %{} = resource_struct_map) do
    resource_type_struct(type, resource_struct_map) <> ".t"
  end

  def resource_open(%Type{module_name: module_name}) do
    """
    #{module_name}.resource.t = e.enif_open_resource_type(env, null, #{module_name}.resource.name, __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
    #{module_name}.Ptr.resource.t = e.enif_open_resource_type(env, null, #{module_name}.Ptr.resource.name, __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
    #{module_name}.Array.resource.t = e.enif_open_resource_type(env, null, #{module_name}.Array.resource.name, __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
    """
  end
end
