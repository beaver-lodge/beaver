defmodule Fizz.CodeGen.Resource do
  def do_resource_type_var("?*anyopaque") do
    "void_ptr"
  end

  def do_resource_type_var("?*const anyopaque") do
    "const_void_ptr"
  end

  def do_resource_type_var(type) do
    type
    |> String.replace("[*c]", "_cptr_")
    |> String.replace(" ", "_")
    |> URI.encode()
    |> String.replace("?", "_nullable_")
    |> String.replace("(", "_")
    |> String.replace(")", "_")
    |> String.replace(",", "__")
    |> String.replace(".", "_")
    |> String.replace("*", "_ptr_")
  end

  def resource_type_var(type) do
    "resource_type_" <> do_resource_type_var(type)
  end

  def resource_type_global(type) do
    """
    // #{type}
    pub var #{resource_type_var(type)}: beam.resource_type = undefined;
    """
  end

  def resource_open(type) do
    """
    // #{type}
    #{resource_type_var(type)} = e.enif_open_resource_type(env, null, "#{resource_type_var(type)}", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
    """
  end
end
