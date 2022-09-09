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

  defmodule AffineMapAcc do
    defstruct exprs: [], contract_index: 0, outer_index: nil
  end

  defp output_rank(rank, contract_axes) do
    # size of another tensor's non-contract axes
    expand_size = rank - length(contract_axes)
    rank + expand_size
  end

  defp gen_input_affine_map(
         %__MODULE__{
           tensor: %{shape: shape},
           batch_axes: [],
           contract_axes: contract_axes
         },
         {maps, outer_index}
       ) do
    rank = tuple_size(shape)

    acc =
      Enum.reduce(
        Range.new(0, rank - 1, 1),
        %AffineMapAcc{outer_index: outer_index},
        fn dim, acc ->
          if dim in contract_axes do
            %{
              acc
              | exprs: acc.exprs ++ [MLIR.AffineMap.dim(acc.contract_index)],
                contract_index: acc.contract_index + 1
            }
          else
            %{
              acc
              | exprs: acc.exprs ++ [MLIR.AffineMap.dim(acc.outer_index)],
                outer_index: acc.outer_index + 1
            }
          end
        end
      )

    output_rank = output_rank(rank, contract_axes)
    {maps ++ [MLIR.AffineMap.create(output_rank, 0, acc.exprs)], acc.outer_index}
  end

  defp gen_output_affine_map(contract_axes_length, rank) when is_integer(rank) do
    out_rank = contract_axes_length + rank

    exprs =
      for dim <- Range.new(0, out_rank - 1, 1), dim >= contract_axes_length do
        MLIR.AffineMap.dim(dim)
      end

    MLIR.AffineMap.create(out_rank, 0, exprs)
  end

  def gen_indexing_maps(
        %__MODULE__{batch_axes: [], contract_axes: contract_axes_a} = a,
        %__MODULE__{batch_axes: [], contract_axes: contract_axes_b} = b,
        c
      )
      when length(contract_axes_a) == length(contract_axes_b) do
    contract_axes_length = length(contract_axes_a)

    {input_maps, _} = Enum.reduce([a, b], {[], 0 + contract_axes_length}, &gen_input_affine_map/2)

    Enum.concat(input_maps, [
      gen_output_affine_map(contract_axes_length, tuple_size(c.shape))
    ])
    |> Enum.map(&MLIR.Attribute.affine_map/1)
    |> Attribute.array()
  end

  def gen_iterator_types(
        %__MODULE__{batch_axes: []},
        %__MODULE__{batch_axes: []}
      ) do
    ~a{["reduction", "parallel", "parallel", "parallel", "parallel"]}
  end

  def gen_output_static_sizes(
        %__MODULE__{batch_axes: [], contract_axes: contract_axes_a},
        %__MODULE__{batch_axes: [], contract_axes: contract_axes_b},
        c
      )
      when length(contract_axes_a) == length(contract_axes_b) do
    Tuple.to_list(c.shape)
  end

  def gen_output_type(
        %__MODULE__{} = a,
        %__MODULE__{} = b,
        c
      ) do
    gen_type(%{c | shape: gen_output_static_sizes(a, b, c) |> List.to_tuple()})
  end
end
