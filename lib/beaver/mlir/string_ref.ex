defmodule Beaver.MLIR.StringRef do
  require Beaver.MLIR.CAPI
  alias Beaver.MLIR.CAPI

  @moduledoc """
  StringRef is very common in LLVM projects. It is a wrapper of a C string. This function will create a Elixir owned C string from a Elixir bitstring and create a StringRef from it. StringRef will keep a reference to the C string to prevent it from being garbage collected by BEAM.
  """
  def create(value) when is_atom(value) do
    value |> Atom.to_string() |> create()
  end

  def create(value) when is_binary(value) do
    c_string = Beaver.Native.c_string(value)

    CAPI.mlirStringRefCreateFromCString(c_string)
    |> Beaver.Native.bag(c_string)
  end

  def extract(%Beaver.MLIR.CAPI.MlirStringRef{} = string_ref) do
    %{ref: ref} =
      string_ref
      |> CAPI.beaverStringRefGetData()

    CAPI.beaver_raw_resource_c_string_to_term_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end
end
