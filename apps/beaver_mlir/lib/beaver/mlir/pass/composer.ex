defmodule Beaver.MLIR.Pass.Composer do
  @moduledoc """
  This module provide functions to compose passes.
  """
  @enforce_keys [:passes, :op]
  defstruct passes: [], op: nil
  import Beaver.MLIR.CAPI
  alias Beaver.MLIR

  def add(composer = %__MODULE__{passes: passes}, pass) do
    %__MODULE__{composer | passes: passes ++ [pass]}
  end

  # TODO: add keyword arguments
  def run!(%__MODULE__{passes: passes, op: op}) do
    ctx = MLIR.Managed.Context.get()
    pm = mlirPassManagerCreate(ctx)

    for pass <- passes do
      mlirPassManagerAddOwnedPass(pm, pass)
    end

    status = mlirPassManagerRun(pm, op)

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure running pass pipeline"
    end

    mlirPassManagerDestroy(pm)
    op
  end
end
