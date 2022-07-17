defmodule Fizz.CodeGen.Function do
  defstruct name: nil, args: [], ret: nil

  def process_type("struct_" <> _ = arg) do
    "c.#{arg}"
  end

  def process_type(arg) do
    String.replace(arg, ~r/\.cimport.+?\./, "c.")
  end

  def do_resource_type_var("?*anyopaque") do
    "void_ptr"
  end

  def do_resource_type_var("?*const anyopaque") do
    "const_void_ptr"
  end

  def do_resource_type_var(type) do
    type
    |> String.replace("[*c]", "_c_ptr_")
    |> String.replace(" ", "_")
    |> URI.encode()
    |> String.replace("?", "_nullable_")
    |> String.replace("(", "_")
    |> String.replace(")", "_")
    |> String.replace(",", "__")
    |> String.replace(".", "_")
    |> String.replace("*", "_pointer_")
  end

  def resource_type_var(type) do
    "resource_type_" <> do_resource_type_var(type)
  end

  def resource_type_global(type) do
    """
    pub var #{resource_type_var(type)}: beam.resource_type = undefined;
    """
  end

  def resource_open(type) do
    """
    #{resource_type_var(type)} = e.enif_open_resource_type(env, null, "#{resource_type_var(type)}", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
    """
  end

  def nif_func_name(func) do
    "fizz_nif_#{func}"
  end

  def nif_declaration(%__MODULE__{name: name, args: args}) do
    """
    e.ErlNifFunc{.name = "#{nif_func_name(name)}", .arity = #{length(args)}, .fptr = #{nif_func_name(name)}, .flags = 0},
    """
  end
end