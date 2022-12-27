defmodule Beaver.MLIR.Value do
  alias Beaver.MLIR.CAPI

  use Kinda.ResourceKind,
    forward_module: Beaver.Native,
    fields: [safe_to_print: true]

  def argument?(%__MODULE__{} = value) do
    CAPI.mlirValueIsABlockArgument(value) |> Beaver.Native.to_term()
  end

  @doc """
  Returns true if the value is a result of an operation.
  """
  def result?(%__MODULE__{} = value) do
    CAPI.mlirValueIsAOpResult(value) |> Beaver.Native.to_term()
  end

  @doc """
  Return the defining op of this value if this value is a result
  """
  def owner(value) do
    if result?(value) do
      {:ok, CAPI.mlirOpResultGetOwner(value)}
    else
      {:error, "not a result"}
    end
  end

  defimpl Inspect do
    def inspect(%{safe_to_print: true} = value, _opts) do
      Beaver.MLIR.to_string(value)
    end

    def inspect(%{safe_to_print: false} = value, _opts) do
      Kernel.inspect(value, structs: false)
    end
  end
end
