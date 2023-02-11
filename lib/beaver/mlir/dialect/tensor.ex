defmodule Beaver.MLIR.Dialect.Tensor do
  @moduledoc """
  This module defines functions for Ops in #{__MODULE__ |> Module.split() |> List.last()} dialect.
  """
  alias Beaver.MLIR.{Attribute, Type}

  use Beaver.MLIR.Dialect,
    dialect: "tensor",
    ops: Beaver.MLIR.Dialect.Registry.ops("tensor")

  def reassociation(list) do
    for grouping <- list do
      grouping
      |> Enum.map(&Attribute.integer(Type.i64(), &1))
      |> Attribute.array()
    end
    |> Attribute.array()
  end

  def reassociation_for_reshape(src, target) do
    Beaver.MLIR.CAPI.beaverGetReassociationIndicesForReshape(src, target)
  end
end
