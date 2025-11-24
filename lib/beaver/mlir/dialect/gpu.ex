defmodule Beaver.MLIR.Dialect.GPU do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect

  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """

  use Beaver.MLIR.Dialect,
    dialect: "gpu",
    ops: Dialect.Registry.ops("gpu")

  @doc """
  Returns the name of the attribute containing the number of buffers located in the workgroup memory.
  ## Examples
    iex> Beaver.MLIR.Dialect.GPU.number_of_buffers_in_workgroup_attributions_attribute_name()
    :workgroup_attributions
  """
  def number_of_buffers_in_workgroup_attributions_attribute_name do
    MLIR.CAPI.beaverGetNumWorkgroupAttributionsAttrName() |> to_string() |> String.to_atom()
  end
end
