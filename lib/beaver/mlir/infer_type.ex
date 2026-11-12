defmodule Beaver.MLIR.InferType do
  @moduledoc """
  Native collection for `InferTypeOpInterface` results without constructing an operation.

  Requests accept `:operation`, `:context`, and optional `:location`, `:operands`, dictionary `:attributes`,
  `:regions`, and `:properties` fields. Properties are supplied as an MLIR attribute. Native code allocates,
  validates and destroys the operation-specific storage within the inference call.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Inference.Request

  @type request :: map() | keyword()

  @spec return_types(request()) :: {:ok, [MLIR.Type.t()]} | {:error, MLIR.diagnostics()}
  def return_types(request), do: Request.infer(request, :beaver_raw_infer_return_types)
end
