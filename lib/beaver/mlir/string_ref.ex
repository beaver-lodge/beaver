alias Beaver.Native

defmodule Beaver.MLIR.StringRef do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.

  call `Kernel.to_string/1` on a `StringRef` to get its content as Elixir bitstring.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  use Kinda.ResourceKind, raw_module: Beaver.MLIR.CAPI.Raw, codec: Native

  @doc """
  Create a Elixir owned null-terminated C string from a Elixir bitstring and create a `StringRef` from it.

  > #### Not really a reference {: .info}
  >
  > A `StringRef` created in Elixir is not a reference to the BEAM binary as argument. Instead, a copy is made. In other words, excessively creating `StringRef` using this function can lead to memory leak.
  """
  def create(value) when is_binary(value),
    do: %__MODULE__{ref: CAPI.beaver_raw_get_string_ref(value)}

  def create(%__MODULE__{} = sr), do: sr

  def create(%MLIR.Identifier{} = id), do: id |> MLIR.CAPI.mlirIdentifierStr()

  def create(value), do: value |> Kernel.to_string() |> create()

  @doc """
  Get the length of the StringRef as Erlang integer.
  """
  def length(%__MODULE__{} = str),
    do: CAPI.beaverStringRefGetLength(str) |> Native.to_term()

  @doc """
  Get the data of the StringRef as an array pointer of unsigned 8-bit integers.
  """
  def data(%__MODULE__{} = str),
    do: CAPI.beaverStringRefGetData(str) |> then(&%Native.Array{ref: &1, element_kind: Native.U8})
end
