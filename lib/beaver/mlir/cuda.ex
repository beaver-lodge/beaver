defmodule Beaver.MLIR.CUDA do
  @moduledoc """
  A minimal Zig-based CUDA runner bootstrap.

  The native side (`beaver_cuda` partition) loads the CUDA Driver API
  (`libcuda`) with `dlopen` at runtime, so Beaver never links against NVIDIA
  libraries and degrades gracefully on machines without a driver. This is the
  first slice of the Zig CUDA runner that will eventually replace the MLIR
  C++ CUDA runtime (`libmlir_cuda_runtime.so`), which is not shipped by the
  eudsl LLVM prebuilts.

  The compilation side keeps using the MLIR C API to produce PTX/cubin
  artifacts (see `Beaver.MLIR.Dialect.GPU.package_binary!/3`); this module only
  covers device discovery for now. Launch/memcpy/cache slices follow.
  """

  alias Beaver.MLIR

  @doc """
  Returns `true` when a CUDA driver is loadable and initializable.

  On machines without `libcuda` (no NVIDIA driver) this returns `false`
  instead of raising.
  """
  @spec available?() :: boolean()
  def available?, do: MLIR.CAPI.beaver_raw_cuda_available()

  @doc """
  Returns `{:ok, count}` with the number of CUDA devices, or `{:error, reason}`
  when the driver is unavailable or a CUDA call fails.
  """
  @spec device_count() :: {:ok, non_neg_integer()} | {:error, String.t()}
  def device_count, do: MLIR.CAPI.beaver_raw_cuda_device_count()

  @doc """
  Returns `{:ok, name}` for the device at `ordinal`, or `{:error, reason}`.
  """
  @spec device_name(integer()) :: {:ok, String.t()} | {:error, String.t()}
  def device_name(ordinal), do: MLIR.CAPI.beaver_raw_cuda_device_name(ordinal)

  @doc """
  Returns `{:ok, {major, minor}}` with the CUDA compute capability of the
  device at `ordinal`, or `{:error, reason}`.
  """
  @spec device_compute_capability(integer()) ::
          {:ok, {non_neg_integer(), non_neg_integer()}} | {:error, String.t()}
  def device_compute_capability(ordinal),
    do: MLIR.CAPI.beaver_raw_cuda_device_compute_capability(ordinal)

  @doc """
  Loads the first `gpu.binary` of an MLIR module into CUDA.

  The PTX assembly of the first `#gpu.object` (as produced by
  `Beaver.MLIR.Dialect.GPU.package_binary!/3` with `format: :isa`) is extracted
  here and loaded natively via `cuModuleLoadData`.

  Returns `{:ok, module_handle}` or `{:error, reason}`.
  """
  @spec load_gpu_binary(MLIR.Module.t()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def load_gpu_binary(%MLIR.Module{} = module) do
    with {:ok, ptx} <- extract_ptx(module) do
      MLIR.CAPI.beaver_raw_cuda_module_load(ptx)
    end
  end

  @doc """
  Returns `{:ok, function_handle}` for the kernel `name` in `module_handle`.
  """
  @spec module_get_function(non_neg_integer(), String.t()) ::
          {:ok, non_neg_integer()} | {:error, String.t()}
  def module_get_function(module_handle, name),
    do: MLIR.CAPI.beaver_raw_cuda_module_get_function(module_handle, name)

  @doc """
  Sets a CUDA function attribute, e.g. the dynamic shared memory size.

  `attribute` is the `CUfunction_attribute` enum value; shared size bytes is
  `0`. Required for Triton kernels that use `global_smem`.
  """
  @spec func_set_attribute(non_neg_integer(), integer(), integer()) :: :ok | {:error, String.t()}
  def func_set_attribute(function_handle, attribute, value),
    do: MLIR.CAPI.beaver_raw_cuda_func_set_attribute(function_handle, attribute, value)

  @doc """
  Unloads a CUDA module handle.
  """
  @spec module_unload(non_neg_integer()) :: :ok | {:error, String.t()}
  def module_unload(module_handle), do: MLIR.CAPI.beaver_raw_cuda_module_unload(module_handle)

  @doc """
  Allocates `size` bytes of device memory. Returns `{:ok, device_ptr}`.
  """
  @spec mem_alloc(non_neg_integer()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def mem_alloc(size), do: MLIR.CAPI.beaver_raw_cuda_mem_alloc(size)

  @doc """
  Frees a device allocation.
  """
  @spec mem_free(non_neg_integer()) :: :ok | {:error, String.t()}
  def mem_free(device_ptr), do: MLIR.CAPI.beaver_raw_cuda_mem_free(device_ptr)

  @doc """
  Blocks until all previously queued work on the current context completes.

  Use after `launch_kernel/4` when timing kernel execution end to end.
  """
  @spec synchronize() :: :ok | {:error, String.t()}
  def synchronize, do: MLIR.CAPI.beaver_raw_cuda_synchronize()

  @doc """
  Copies host `data` into device memory at `device_ptr`.
  """
  @spec memcpy_htod(non_neg_integer(), binary()) :: :ok | {:error, String.t()}
  def memcpy_htod(device_ptr, data), do: MLIR.CAPI.beaver_raw_cuda_memcpy_htod(device_ptr, data)

  @doc """
  Copies `size` bytes from device memory at `device_ptr` into a host binary.
  """
  @spec memcpy_dtoh(non_neg_integer(), non_neg_integer()) ::
          {:ok, binary()} | {:error, String.t()}
  def memcpy_dtoh(device_ptr, size),
    do: MLIR.CAPI.beaver_raw_cuda_memcpy_dtoh(device_ptr, size)

  @doc """
  Launches a kernel.

  `grid`/`block` are `{x, y, z}` tuples. `args` is a list of kernel parameters:

    * `{:f32, value}` - 32-bit float, padded to an 8-byte slot
    * `{:i64, value}` - 64-bit integer
    * `{:ptr, device_ptr}` - device pointer

  Slots are packed in order, matching the aligned `.param` layout of NVVM
  kernels. Returns `:ok` or `{:error, reason}`.
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

    MLIR.CAPI.beaver_raw_cuda_launch_kernel(
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
    with {:ok, op} <- fetch_gpu_binary(module),
         {:ok, objects} <- MLIR.Operation.fetch(op, "objects"),
         {:ok, object} <- MLIR.Attribute.fetch(objects, 0) do
      extract_assembly(object)
    else
      _ -> {:error, "no gpu.binary with objects found"}
    end
  end

  defp fetch_gpu_binary(module) do
    case module
         |> MLIR.Module.body()
         |> Beaver.Walker.operations()
         |> Enum.find(&(MLIR.Operation.name(&1) == "gpu.binary")) do
      %MLIR.Operation{} = op -> {:ok, op}
      _ -> {:error, :not_found}
    end
  end

  defp extract_assembly(object) do
    case Regex.run(~r/assembly = "((?:[^"\\]|\\.)*)"/, MLIR.to_string(object)) do
      [_, escaped] -> {:ok, unescape_mlir_string(escaped)}
      _ -> {:error, "gpu.object has no assembly"}
    end
  end

  # MLIR prints string attrs with C-style escapes: `\\` for backslash, `\"`
  # for a quote, and `\XX` for non-printable bytes (e.g. `\0A` for newline).
  defp unescape_mlir_string(escaped), do: do_unescape(String.graphemes(escaped), [])

  defp do_unescape([], acc), do: acc |> Enum.reverse() |> IO.iodata_to_binary()

  defp do_unescape(["\\", "\\" | rest], acc), do: do_unescape(rest, ["\\" | acc])
  defp do_unescape(["\\", "\"" | rest], acc), do: do_unescape(rest, ["\"" | acc])

  defp do_unescape(["\\", h1, h2 | rest], acc)
       when h1 in ~w(0 1 2 3 4 5 6 7 8 9 a b c d e f A B C D E F) and
              h2 in ~w(0 1 2 3 4 5 6 7 8 9 a b c d e f A B C D E F) do
    do_unescape(rest, [<<String.to_integer(h1 <> h2, 16)>> | acc])
  end

  defp do_unescape([char | rest], acc), do: do_unescape(rest, [char | acc])
end
