defmodule Beaver.MLIR.StringRef do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """
  alias Beaver.MLIR.CAPI
  use Kinda.ResourceKind, forward_module: Beaver.Native

  @doc """
  Create a Elixir owned null-terminated C string from a Elixir bitstring and create a `StringRef` from it.

  > #### Not really a reference {: .info}
  >
  > A `StringRef` created in Elixir is not a reference to the BEAM binary as argument. Instead, a copy is made. In other words, excessively creating `StringRef` using this function can lead to memory leak.
  """
  def create(value) when is_binary(value) do
    %__MODULE__{ref: CAPI.beaver_raw_get_string_ref(value)}
  end

  def create(%__MODULE__{} = sr) do
    sr
  end

  def create(value) do
    value |> Kernel.to_string() |> create()
  end

  @doc """
  Get the length of the StringRef as Erlang integer.
  """
  def length(%__MODULE__{} = str) do
    CAPI.beaverStringRefGetLength(str) |> Beaver.Native.to_term()
  end

  @doc """
  Get the data of the StringRef as an array pointer of unsigned 8-bit integers.
  """
  def data(%__MODULE__{} = str) do
    CAPI.beaverStringRefGetData(str)
    |> then(&%Beaver.Native.Array{ref: &1, element_kind: Beaver.Native.U8})
  end
end
