defmodule Exotic do
  @moduledoc """
  Documentation for `Exotic`.
  """
  alias Exotic.{Function, Value, Library, NIF}

  def extract_ref(%{ref: ref}) do
    ref
  end

  defp get_managed_libarary(module) do
    managed_module = Module.concat(module, Managed)
    apply(managed_module, :value, [])
  end

  def call!(_, _, _args \\ :default)

  def call!(%Library{ref: lib_ref, functions: functions}, func_name, args)
      when is_atom(func_name) do
    arg_values =
      args
      |> Enum.map(&Value.get/1)

    arg_refs = arg_values |> Enum.map(&extract_ref/1)

    holdings =
      arg_values
      |> Enum.map(&Map.get(&1, :holdings))
      |> Enum.reduce(MapSet.new(), &MapSet.union(&2, &1))

    %Function{ref: func_ref, def: %Function.Definition{return_type: return_type}} =
      functions
      |> Map.fetch!({func_name, length(args)})

    result_ref = NIF.call_func(lib_ref, func_ref, arg_refs)
    struct!(Value, %{ref: result_ref, type: return_type, holdings: holdings})
  end

  # TODO: benchmark to see if it faster to create function wrapper each time or cache it
  def call!(
        %Library{ref: lib_ref},
        func_def = %Exotic.Function.Definition{return_type: return_type, arg_types: arg_types},
        args
      ) do
    arg_values =
      Enum.zip(arg_types, args)
      |> Enum.map(fn {t, v} -> Value.get(t, v) end)

    arg_refs = arg_values |> Enum.map(&extract_ref/1)

    holdings =
      arg_values
      |> Enum.map(&Map.get(&1, :holdings))
      |> Enum.reduce(MapSet.new(), &MapSet.union(&2, &1))

    %Function{ref: func_ref} = Function.get(func_def)
    result_ref = NIF.call_func(lib_ref, func_ref, arg_refs)
    struct!(Value, %{ref: result_ref, type: return_type, holdings: holdings})
  end

  def call!(module, func_name, args)
      when is_atom(func_name) and is_atom(module) do
    get_managed_libarary(module) |> call!(func_name, args)
  end

  def call!(module, func_def = %Exotic.Function.Definition{}, args)
      when is_atom(module) do
    get_managed_libarary(module) |> call!(func_def, args)
  end

  def call(lib, fun, args \\ []) do
    {:ok, call!(lib, fun, args)}
  end

  def load(module, path) do
    {:ok, load!(module, path)}
  end

  def load(path) do
    {:ok, load!(path)}
  end

  defp get_functions(module) do
    got =
      apply(Module.concat(module, Meta), :native_definitions, [])
      |> Map.new(fn {k, v} -> {k, Function.get(v)} end)

    if Enum.empty?(got), do: raise("no native functions found in module #{module}")
    got
  end

  def load!(module, path) when is_atom(module) and is_binary(path) do
    ref = NIF.get_lib(path)
    %Library{ref: ref, path: path, functions: get_functions(module), id: module}
  end

  def load!(module, loaded_module)
      when is_atom(module) and is_atom(loaded_module) and not is_nil(loaded_module) do
    %Library{ref: ref, path: path} = get_managed_libarary(loaded_module)
    %Library{ref: ref, path: path, functions: get_functions(module), id: module}
  end

  def load!(path) when is_binary(path) do
    ref = NIF.get_lib(path)
    %Library{ref: ref, path: path, functions: %{}, id: :unknown}
  end

  def load!(module) when is_atom(module) do
    # TODO: support paths
    get_managed_libarary(module)
  end
end
