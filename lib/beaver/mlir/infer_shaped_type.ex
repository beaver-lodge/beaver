defmodule Beaver.MLIR.InferShapedType do
  @moduledoc """
  Native collection for `InferShapedTypeOpInterface` components.

  Ranked shapes expose dynamic dimensions as `:dynamic`; unranked results use
  `:unranked`. Element types and optional encoding attributes remain
  owned by the request context. As with `Beaver.MLIR.InferType`, non-`nil`
  operation properties are rejected because the upstream ABI is an untyped
  `void *`.
  """

  alias Beaver.MLIR
  alias Beaver.MLIR.Inference.Request

  defmodule Component do
    @moduledoc "A shaped inference result owned by its originating MLIR context."
    @enforce_keys [:shape, :element_type, :encoding]
    defstruct [:shape, :element_type, :encoding]

    @type t :: %__MODULE__{
            shape: [non_neg_integer() | :dynamic] | :unranked,
            element_type: MLIR.Type.t() | nil,
            encoding: MLIR.Attribute.t() | nil
          }
  end

  @type request :: map() | keyword()

  @spec return_components(request()) ::
          {:ok, [Component.t()]} | {:error, MLIR.diagnostics()}
  def return_components(request) do
    request
    |> Request.raw_arguments()
    |> then(&apply(MLIR.CAPI, :beaver_raw_infer_return_type_components, &1))
    |> Beaver.Native.check!()
    |> case do
      {components, _diagnostics} when is_list(components) ->
        {:ok, Enum.map(components, &normalize_component/1)}

      {:error, diagnostics} ->
        {:error, diagnostics}
    end
  end

  defp normalize_component({shape, element_type, encoding}) do
    %Component{
      shape: normalize_shape(shape),
      element_type: normalize_optional_handle(element_type),
      encoding: normalize_optional_handle(encoding)
    }
  end

  defp normalize_optional_handle(nil), do: nil
  defp normalize_optional_handle(handle), do: Beaver.Native.check!(handle)

  defp normalize_shape(:unranked), do: :unranked

  defp normalize_shape(shape) do
    Enum.map(shape, &MLIR.ShapedType.cast_dynamic_magic_number(&1, :size))
  end
end
