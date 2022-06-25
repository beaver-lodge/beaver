defmodule Beaver.Nx do
  alias Beaver.MLIR
  import Beaver.MLIR.Sigils
  import MLIR.{Transforms, Conversion}
  alias Beaver.MLIR.ExecutionEngine.MemRefDescriptor

  @enforce_keys [:memref]
  defstruct [:memref]

  @behaviour Nx.Backend
  @moduledoc """
  Documentation for `Beaver.Nx`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> Beaver.Nx.hello()
      :world

  """
  def hello do
    :world
  end

  alias Nx.Tensor, as: T
  alias __MODULE__, as: B

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, backend_options) do
    shape = Tuple.to_list(shape)

    memref =
      MemRefDescriptor.create(
        binary,
        shape,
        MemRefDescriptor.dense_strides(shape)
      )

    {memref} |> Beaver.Nx.MemrefAllocator.add()
    put_in(tensor.data, %B{memref: memref})
  end

  @impl true
  def to_binary(%T{shape: shape, data: %B{memref: memref}, type: {_, size}}, limit) do
    memref
    |> Exotic.Value.fetch(
      MemRefDescriptor.struct_fields(tuple_size(shape)),
      :aligned
    )
    |> Exotic.Value.Ptr.read_as_binary(limit * div(size, 8))
  end

  @impl true
  def backend_transfer(
        %T{shape: shape, data: %B{memref: memref}, type: {_, size}} = tensor,
        backend,
        backend_options
      ) do
    if backend == __MODULE__ do
      tensor
    else
      binary_len = Enum.reduce(Tuple.to_list(shape), 1, &*/2) * div(size, 8)

      binary =
        memref
        |> Exotic.Value.fetch(
          MemRefDescriptor.struct_fields(tuple_size(shape)),
          :aligned
        )
        |> Exotic.Value.Ptr.read_as_binary(binary_len)

      with :ok <- Beaver.Nx.MemrefAllocator.delete(memref) do
        Nx.BinaryBackend.from_binary(tensor, binary, backend_options)
      else
        :already_deallocated -> raise "called on deleted or donated buffer"
      end
    end
  end

  @impl true
  def backend_deallocate(%T{data: %B{memref: memref}}) do
    memref |> Beaver.Nx.MemrefAllocator.delete()
  end
end
