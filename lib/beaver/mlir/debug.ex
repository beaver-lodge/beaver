defmodule Beaver.MLIR.Debug do
  @moduledoc """
  Configure MLIR debug options.
  """
  alias Beaver.MLIR
  import MLIR.CAPI

  def enable_global_debug(enable \\ true) do
    mlirEnableGlobalDebug(enable)
  end

  def disable_global_debug() do
    enable_global_debug(false)
  end

  @doc """
  Check if global debugging is enabled.

  ## Examples

      iex> Beaver.MLIR.Debug.global_debug_enabled?()
      false
  """
  def global_debug_enabled?() do
    mlirIsGlobalDebugEnabled() |> Beaver.Native.to_term()
  end

  @doc """
  Set the current debug type.

  Note: Global debug must be enabled for any output to be produced.

  ## Examples

      iex> Beaver.MLIR.Debug.set_debug_type("pass-manager")
      :ok
  """
  def set_debug_type(type) when is_binary(type) do
    type_str = MLIR.StringRef.create(type) |> MLIR.StringRef.data()
    mlirSetGlobalDebugType(type_str)
  end

  @doc """
  Set multiple debug types.

  Note: Global debug must be enabled for any output to be produced.

  ## Examples

      iex> Beaver.MLIR.Debug.set_debug_types(["pass-manager", "transform"])
      :ok
  """
  def set_debug_types(types) when is_list(types) do
    types
    |> Enum.map(&MLIR.StringRef.create/1)
    |> Beaver.Native.array(MLIR.StringRef)
    |> beaverSetGlobalDebugTypes(length(types))
  end

  @doc """
  Check if a specific debug type is currently enabled.

  ## Examples

      iex> Beaver.MLIR.Debug.is_current_debug_type?("pass-manager")
      true
  """
  def is_current_debug_type?(type) when is_binary(type) do
    type_str = MLIR.StringRef.create(type) |> MLIR.StringRef.data()
    mlirIsCurrentDebugType(type_str) |> Beaver.Native.to_term()
  end
end
