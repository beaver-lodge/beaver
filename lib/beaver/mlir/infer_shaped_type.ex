defmodule Beaver.MLIR.InferShapedType do
  @moduledoc """
  Native collection for `InferShapedTypeOpInterface` components.

  Ranked shapes expose dynamic dimensions as `:dynamic`; unranked results use
  `:unranked`. Element types and optional encoding attributes remain
  owned by the request context. Operation properties use the same scoped,
  attribute-based conversion as `Beaver.MLIR.InferType`.
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
  def return_components(request),
    do: Request.infer(request, :beaver_raw_infer_return_type_components, &normalize_component/1)

  defp normalize_component({shape, element_type, encoding}),
    do: %Component{
      shape: normalize_shape(shape),
      element_type: Beaver.Native.check!(element_type),
      encoding: Beaver.Native.check!(encoding)
    }

  defp normalize_shape(:unranked), do: :unranked

  defp normalize_shape(shape),
    do: Enum.map(shape, &MLIR.ShapedType.cast_dynamic_magic_number(&1, :size))
end
