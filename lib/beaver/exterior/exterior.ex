defmodule Beaver.Exterior do
  @moduledoc """
  Behavior to build and register a dialect not included in LLVM mono repo. It is called "Exterior" not "Extension" or "External" to avoid confusing it with various LLVM/MLIR concepts of extensions.
  """
  alias Beaver.MLIR

  @doc """
  Register a register to a MLIR context
  """
  @callback register_dialect(MLIR.Context.t()) :: :ok | {:error, String.t()}

  def register_all(ctx) do
    # TODO: get it from app config or other mechanism
    for dialect <- [] do
      :ok = dialect.register_dialect(ctx)
    end
  end
end
