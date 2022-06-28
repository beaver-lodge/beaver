defmodule Beaver.Nx do
  @moduledoc """
  `Beaver.Nx` is a MLIR backend for the `Nx`. It mainly targets TOSA dialect.
  """

  alias Beaver.MLIR.ExecutionEngine.MemRefDescriptor

  @enforce_keys [:memref]
  defstruct [:memref]

  @behaviour Nx.Backend

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
  def to_binary(%T{shape: _shape, data: %B{memref: memref} = tensor}, limit) do
    MemRefDescriptor.read_as_binary(memref, limit * div(element_size(tensor), 8))
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1

    tensor
    |> to_binary(min(limit, Nx.size(tensor)))
    |> then(&Nx.Backend.inspect(tensor, &1, inspect_opts))
  end

  def element_size(%T{type: {_, size}}), do: size

  @impl true
  def backend_copy(tensor, Nx.Tensor, backend_options) do
    backend_copy(tensor, Nx.BinaryBackend, backend_options)
  end

  # TODO: Support direct transfers without going through Elixir
  def backend_copy(
        %T{shape: shape, data: %B{memref: memref}} = tensor,
        backend,
        backend_options
      ) do
    binary_len = Enum.reduce(Tuple.to_list(shape), 1, &*/2) * div(element_size(tensor), 8)

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

  @impl true
  def multiply(out, l, h) do
    out = Nx.to_template(out)

    expr_fun = fn t1, t2 ->
      Nx.Defn.Expr.multiply(out, t1, t2)
    end

    options = [force: true]
    Nx.Defn.jit(expr_fun, [l, h], Keyword.put(options, :compiler, Beaver.Nx.Compiler))
  end

  @doc """
  Create a new tensor of null ptr memref. This should be used as as the return tensor of JIT function.
  """
  def tensor_of_null_memref(%T{shape: shape, type: _type} = tensor) do
    shape = Tuple.to_list(shape)

    memref =
      MemRefDescriptor.create(
        shape,
        MemRefDescriptor.dense_strides(shape)
      )

    # TODO: delete the allocated ptr when this kind of tensor is deallocated by Nx
    {memref} |> Beaver.Nx.MemrefAllocator.add()
    put_in(tensor.data, %B{memref: memref})
  end

  def tensor_of_null_memref(tuple) when is_tuple(tuple) do
    for t <- tuple |> Tuple.to_list() do
      tensor_of_null_memref(t)
    end
    |> List.to_tuple()
  end
end
