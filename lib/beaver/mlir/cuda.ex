alias Beaver.MLIR.{CAPI, Operation}

defmodule Beaver.MLIR.CUDA do
  @moduledoc """
  CUDA driver discovery, module loading, memory copies and kernel launches.

  The Zig runner loads `libcuda` dynamically, without linking NVIDIA libraries, and reports unavailable drivers
  gracefully. Compilation remains separate; see `Beaver.MLIR.Dialect.GPU.package_binary!/3` for PTX/cubin packaging.
  """
  alias Beaver.MLIR
  alias Beaver.MLIR.Dialect.GPU

  @doc """
  Returns `true` when a CUDA driver is loadable and initializable.

  On machines without `libcuda` (no NVIDIA driver) this returns `false` instead of raising.
  """
  @spec available?() :: boolean()
  def available?, do: CAPI.beaver_raw_cuda_available()

  @doc """
  Returns `{:ok, count}` with the number of CUDA devices, or `{:error, reason}` when the driver is unavailable or a
  CUDA call fails.
  """
  @spec device_count() :: {:ok, non_neg_integer()} | {:error, String.t()}
  def device_count, do: CAPI.beaver_raw_cuda_device_count()
  @doc "Returns `{:ok, name}` for the device at `ordinal`, or `{:error, reason}`."
  @spec device_name(integer()) :: {:ok, String.t()} | {:error, String.t()}
  def device_name(ordinal), do: CAPI.beaver_raw_cuda_device_name(ordinal)

  @doc """
  Returns `{:ok, {major, minor}}` with the CUDA compute capability of the device at `ordinal`, or `{:error,
  reason}`.
  """
  @spec device_compute_capability(integer()) :: {:ok, {non_neg_integer(), non_neg_integer()}} | {:error, String.t()}
  def device_compute_capability(ordinal), do: CAPI.beaver_raw_cuda_device_compute_capability(ordinal)

  @doc """
  Loads the first `gpu.binary` of an MLIR module into CUDA.

  The PTX assembly of the first `#gpu.object` (as produced by `Beaver.MLIR.Dialect.GPU.package_binary!/3` with
  `format: :isa`) is extracted here and loaded natively via `cuModuleLoadData`.

  Returns `{:ok, module_handle}` or `{:error, reason}`.
  """
  @spec load_gpu_binary(MLIR.Module.t()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def load_gpu_binary(%MLIR.Module{} = module) do
    with {:ok, ptx} <- extract_ptx(module) do
      CAPI.beaver_raw_cuda_module_load(ptx)
    end
  end

  @doc "Returns `{:ok, function_handle}` for the kernel `name` in `module_handle`."
  @spec module_get_function(non_neg_integer(), String.t()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def module_get_function(module_handle, name), do: CAPI.beaver_raw_cuda_module_get_function(module_handle, name)

  @doc """
  Sets a CUDA function attribute, e.g. the dynamic shared memory size.

  `attribute` is the `CUfunction_attribute` enum value. To opt into larger dynamic shared memory use
  `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` (`8`), not the read-only static shared-size attribute (`0`).
  """
  @spec func_set_attribute(non_neg_integer(), integer(), integer()) :: :ok | {:error, String.t()}
  def func_set_attribute(function_handle, attribute, value),
    do: CAPI.beaver_raw_cuda_func_set_attribute(function_handle, attribute, value)

  @doc "Unloads a CUDA module handle."
  @spec module_unload(non_neg_integer()) :: :ok | {:error, String.t()}
  def module_unload(module_handle), do: CAPI.beaver_raw_cuda_module_unload(module_handle)
  @doc "Allocates `size` bytes of device memory. Returns `{:ok, device_ptr}`."
  @spec mem_alloc(non_neg_integer()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def mem_alloc(size), do: CAPI.beaver_raw_cuda_mem_alloc(size)
  @doc "Frees a device allocation."
  @spec mem_free(non_neg_integer()) :: :ok | {:error, String.t()}
  def mem_free(device_ptr), do: CAPI.beaver_raw_cuda_mem_free(device_ptr)

  @doc """
  Blocks until all previously queued work on the current context completes.

  Use after `launch_kernel/4` when timing kernel execution end to end.
  """
  @spec synchronize() :: :ok | {:error, String.t()}
  def synchronize, do: CAPI.beaver_raw_cuda_synchronize()
  @doc "Copies host `data` into device memory at `device_ptr`."
  @spec memcpy_htod(non_neg_integer(), binary()) :: :ok | {:error, String.t()}
  def memcpy_htod(device_ptr, data), do: CAPI.beaver_raw_cuda_memcpy_htod(device_ptr, data)
  @doc "Copies `size` bytes from device memory at `device_ptr` into a host binary."
  @spec memcpy_dtoh(non_neg_integer(), non_neg_integer()) :: {:ok, binary()} | {:error, String.t()}
  def memcpy_dtoh(device_ptr, size), do: CAPI.beaver_raw_cuda_memcpy_dtoh(device_ptr, size)

  @doc """
  Copies device allocations without host staging, ordered on the default stream.

  The caller retains both allocations until completion. Use `synchronize/0` when a host-visible completion boundary
  is required. Overlap is unsupported.
  """
  @spec memcpy_dtod(non_neg_integer(), non_neg_integer(), non_neg_integer()) :: :ok | {:error, String.t()}
  def memcpy_dtod(destination, source, size), do: CAPI.beaver_raw_cuda_memcpy_dtod(destination, source, size)

  @doc """
  Launches a kernel.

  `grid`/`block` are `{x, y, z}` tuples. `args` is a list of kernel parameters:

  * `{:f32, value}` - 32-bit float, padded to an 8-byte slot * `{:i64, value}` - 64-bit integer * `{:ptr,
  device_ptr}` - device pointer

  Slots are packed in order, matching the aligned `.param` layout of NVVM kernels. Returns `:ok` or `{:error,
  reason}`.
  """
  @spec launch_kernel(
          non_neg_integer(),
          {non_neg_integer(), non_neg_integer(), non_neg_integer()},
          {non_neg_integer(), non_neg_integer(), non_neg_integer()},
          [{:f32, float()} | {:i64, integer()} | {:ptr, non_neg_integer()}]
        ) :: :ok | {:error, String.t()}
  def launch_kernel(function_handle, grid, block, args, opts \\ []) do
    {grid_x, grid_y, grid_z} = grid
    {block_x, block_y, block_z} = block

    CAPI.beaver_raw_cuda_launch_kernel(
      function_handle,
      grid_x,
      grid_y,
      grid_z,
      block_x,
      block_y,
      block_z,
      Keyword.get(opts, :shared_mem, 0),
      pack_args(args)
    )
  end

  defp pack_args(args) do
    args
    |> Enum.map(fn
      {:f32, value} -> <<value::float-32-little>> <> <<0::32>>
      {:i64, value} -> <<value::signed-64-little>>
      {:ptr, value} -> <<value::unsigned-64-little>>
    end)
    |> IO.iodata_to_binary()
  end

  defp extract_ptx(module) do
    operations = module |> MLIR.Module.body() |> Beaver.Walker.operations()

    with %Operation{} = op <- Enum.find(operations, &(Operation.name(&1) == "gpu.binary")),
         {:ok, objects} <- Operation.fetch(op, "objects"),
         {:ok, object} <- MLIR.Attribute.fetch(objects, 0) do
      GPU.object_data(object)
    else
      _ -> {:error, "no gpu.binary with objects found"}
    end
  end
end
