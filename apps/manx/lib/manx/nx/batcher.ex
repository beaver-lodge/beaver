defmodule Manx.Nx.Batcher do
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.TOSA
  require Beaver.MLIR
  import MLIR.Sigils
  import Beaver, only: :macros
  alias Manx.Defn.Env
  alias MLIR.{Type, Attribute}
  import Manx.Type
  defstruct [:tensor, :value, :contract_axes, :batch_axes]

  def from_args(
        [
          %Nx.Tensor{} = a,
          contract_axes1,
          batch_axes1,
          %Nx.Tensor{} = b,
          contract_axes2,
          batch_axes2
        ],
        a_value,
        b_value
      ) do
    batched_a = %Manx.Nx.Batcher{
      tensor: a,
      value: a_value,
      contract_axes: contract_axes1,
      batch_axes: batch_axes1
    }

    batched_b = %Manx.Nx.Batcher{
      tensor: b,
      value: b_value,
      contract_axes: contract_axes2,
      batch_axes: batch_axes2
    }

    {batched_a, batched_b}
  end

  def get_batch_type(%Nx.Tensor{shape: {dim0, dim1}} = t) do
    Manx.Type.gen_type(%{t | shape: {1, dim0, dim1}})
  end

  def get_batch_type(%Nx.Tensor{shape: shape} = t) when tuple_size(shape) == 3 do
    Manx.Type.gen_type(t)
  end

  @doc """
  transform batch axes to 0
  """
  def gen_batched_transpose(
        %Env{block: block},
        %__MODULE__{tensor: %{shape: shape} = t, value: value}
      )
      when tuple_size(shape) == 2 do
    mlir block: block do
      Tensor.expand_shape(value, reassociation: Tensor.reassociation([[0, 1], [2]])) >>>
        get_batch_type(t)
    end
  end

  def gen_batched_transpose(
        %Env{block: block},
        %__MODULE__{tensor: %{shape: shape}, value: value}
      )
      when tuple_size(shape) == 3 do
    mlir block: block do
      value
    end
  end

  def gen_batched_transpose(
        %Env{block: block},
        %__MODULE__{tensor: tensor, value: value, batch_axes: [batch_axis]}
      ) do
    mlir block: block do
      perms =
        Range.new(0, tuple_size(tensor.shape) - 1, 1)
        |> Enum.to_list()
        |> List.update_at(batch_axis, 0)
        |> List.update_at(0, batch_axis)

      perms = for perm <- perms, do: Attribute.integer(Type.i32(), perm)

      perms =
        Attribute.dense_elements(
          perms,
          Type.ranked_tensor([2], Type.i32())
        )

      TOSA.transpose(value, perms: perms) >>> Type.unranked_tensor(gen_type(tensor.type))
    end
  end

  def gen_batched_collapse(
        %Env{block: block},
        value,
        t
      ) do
    mlir block: block do
      Tensor.collapse_shape(value, reassociation: Tensor.reassociation([[0, 1], [2]])) >>>
        Manx.Type.gen_type(t)
    end
  end

  defp gen_affine_map(%__MODULE__{
         tensor: %{shape: shape},
         batch_axes: [],
         contract_axes: contract_axes
       }) do
    rank = tuple_size(shape)

    exprs =
      for dim <- Range.new(0, rank - 1, 1), dim not in contract_axes do
        MLIR.AffineMap.dim(dim)
      end

    MLIR.AffineMap.create(rank, 0, exprs)
  end

  def gen_indexing_maps(
        %__MODULE__{batch_axes: []} = a,
        %__MODULE__{batch_axes: []} = b
      ) do
    [a, b]
    |> Enum.map(&gen_affine_map/1)
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  def gen_iterator_types(
        %__MODULE__{batch_axes: []},
        %__MODULE__{batch_axes: []}
      ) do
  end
end
