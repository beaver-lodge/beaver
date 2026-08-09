defmodule Beaver.MLIR.InferType do
  @moduledoc """
  Native collection for `InferTypeOpInterface` results without constructing an
  operation.

  Requests accept `:operation`, `:context`, and optional `:location`,
  `:operands`, dictionary `:attributes`, `:regions`, and `:properties` fields.
  The upstream properties argument is an untyped `void *`; this API therefore
  rejects every non-`nil` value until an operation-specific ownership contract
  exists.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Inference.Request

  @type request :: map() | keyword()

  @spec return_types(request()) :: {:ok, [MLIR.Type.t()]} | {:error, MLIR.diagnostics()}
  def return_types(request) do
    request
    |> Request.raw_arguments()
    |> then(&apply(MLIR.CAPI, :beaver_raw_infer_return_types, &1))
    |> Beaver.Native.check!()
    |> case do
      {types, _diagnostics} when is_list(types) ->
        {:ok, Enum.map(types, &Beaver.Native.check!/1)}

      {:error, diagnostics} ->
        {:error, diagnostics}
    end
  end
end
