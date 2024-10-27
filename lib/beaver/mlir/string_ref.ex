defmodule Beaver.MLIR.StringRef do
  @moduledoc """
  This module defines functions working with MLIR #{__MODULE__ |> Module.split() |> List.last()}.
  """

  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind, forward_module: Beaver.Native

  @doc """
  Create a Elixir owned null-terminated C string from a Elixir bitstring and create a `StringRef` from it.

  > Note: A `StringRef` will not reference the original BEAM binary.
  > Instead, it will reference a copy of the binary and owns it.
  > In other words, excessively creating `StringRef` using this function can lead to memory leak.
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
  Converts an `StringRef` to a string.
  """
  def to_string(%__MODULE__{ref: ref}) do
    CAPI.beaver_raw_string_ref_to_binary(ref)
  end

  def length(%__MODULE__{} = str) do
    CAPI.beaverStringRefGetLength(str) |> Beaver.Native.to_term()
  end

  def data(%__MODULE__{} = str) do
    CAPI.beaverStringRefGetData(str)
    |> then(&%Beaver.Native.Array{ref: &1, element_kind: Beaver.Native.U8})
  end

  defimpl String.Chars do
    defdelegate to_string(mlir), to: Beaver.MLIR.StringRef
  end
end
