defmodule Beaver.MLIR.StringRef do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """

  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind,
    forward_module: Beaver.Native

  @doc """
  Create a Elixir owned null-terminated C string from a Elixir bitstring and create a `StringRef` from it.

  > Note: A `StringRef` will not reference the original BEAM binary.
  > Instead, it will reference a copy of the binary and owns it.
  > In other words, excessively creating `StringRef` using this function can lead to memory leak.
  """
  def create(value) when is_atom(value) do
    value |> Atom.to_string() |> create()
  end

  def create(value) when is_binary(value) do
    %__MODULE__{ref: CAPI.beaver_raw_get_string_ref(value)}
  end

  @doc """
  Converts an `StringRef` to a string.
  """
  def to_string(%__MODULE__{} = string_ref) do
    %{ref: ref} =
      string_ref
      |> CAPI.beaverStringRefGetData()

    CAPI.beaver_raw_resource_c_string_to_term_charlist(ref)
    |> Beaver.Native.check!()
    |> List.to_string()
  end
end
