defmodule Fizz.CodeGen.Function do
  defstruct name: nil, args: [], ret: nil
  import Fizz.CodeGen.Resource

  def process_type("struct_" <> _ = arg) do
    "c.#{arg}"
  end

  def process_type(arg) do
    String.replace(arg, ~r/\.cimport.+?\./, "c.")
  end

  def array_maker_name(type) do
    "fizz_nif_get_resource_array_#{resource_type_var(type)}"
  end

  def ptr_maker_name(type) do
    "fizz_nif_get_resource_ptr_#{resource_type_var(type)}"
  end

  def primitive_maker_name(type) do
    "fizz_nif_get_primitive_from_#{resource_type_var(type)}"
  end

  def resource_maker_name(type) do
    "fizz_nif_get_resource_of_#{resource_type_var(type)}"
  end

  def nif_func_name(func) do
    "fizz_nif_#{func}"
  end
end
