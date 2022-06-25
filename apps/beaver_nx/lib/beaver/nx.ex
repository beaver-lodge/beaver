defmodule Beaver.Nx do
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
  def constant(out, constant, backend_options) do
    binary_tensor = Nx.BinaryBackend.constant(out, constant, [])
    Nx.BinaryBackend.backend_transfer(binary_tensor, __MODULE__, backend_options)
  end

  @impl true
  def from_binary(%T{shape: shape, type: _type} = tensor, binary, _backend_options) do
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
  def to_binary(%T{shape: _shape, data: %B{memref: memref}, type: {_, size}}, limit) do
    MemRefDescriptor.read_as_binary(memref, limit * div(size, 8))
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
  end

  @impl true
  def backend_copy(tensor, Nx.Tensor, backend_options) do
    backend_copy(tensor, Nx.BinaryBackend, backend_options)
  end

  # TODO: Support direct transfers without going through Elixir
  def backend_copy(
        %T{shape: shape, data: %B{memref: memref}, type: {_, size}} = tensor,
        backend,
        backend_options
      ) do
    binary_len = Enum.reduce(Tuple.to_list(shape), 1, &*/2) * div(size, 8)

    backend.from_binary(
      tensor,
      MemRefDescriptor.read_as_binary(memref, binary_len),
      backend_options
    )
  end

  @impl true
  def backend_transfer(
        %T{data: %B{memref: memref}} = tensor,
        backend,
        backend_options
      ) do
    if backend == __MODULE__ do
      # TODO: support tensor on device memory like CUDA
      tensor
    else
      tensor = backend_copy(tensor, backend, backend_options)

      with :ok <- Beaver.Nx.MemrefAllocator.delete(memref) do
        tensor
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
