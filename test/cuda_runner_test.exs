defmodule CudaRunnerTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR.CUDA

  @moduletag :cuda

  test "device copies preserve offsets, guards, and independent snapshot storage" do
    if CUDA.available?() do
      {:ok, source} = CUDA.mem_alloc(64)
      on_exit(fn -> CUDA.mem_free(source) end)
      {:ok, destination} = CUDA.mem_alloc(64)
      on_exit(fn -> CUDA.mem_free(destination) end)
      input = :binary.list_to_bin(Enum.to_list(0..63))
      assert :ok = CUDA.memcpy_htod(source, input)
      assert :ok = CUDA.memcpy_htod(destination, :binary.copy(<<255>>, 64))
      assert :ok = CUDA.memcpy_dtod(destination + 8, source + 16, 32)
      assert :ok = CUDA.synchronize()
      assert :ok = CUDA.memcpy_htod(source, :binary.copy(<<0>>, 64))
      assert {:ok, output} = CUDA.memcpy_dtoh(destination, 64)

      assert output == :binary.copy(<<255>>, 8) <> binary_part(input, 16, 32) <> :binary.copy(<<255>>, 24)

      assert :ok = CUDA.memcpy_dtod(destination, source, 0)
      assert {:error, _} = CUDA.memcpy_dtod(source + 1, source, 32)
    else
      assert {:error, _} = CUDA.memcpy_dtod(1, 2, 1)
    end
  end

  test "device discovery runs or degrades gracefully" do
    case CUDA.available?() do
      true ->
        assert {:ok, count} = CUDA.device_count()
        assert count >= 1
        assert {:ok, name} = CUDA.device_name(0)
        assert is_binary(name) and name != ""

      false ->
        # No NVIDIA driver: the runner must not crash or raise.
        assert {:error, reason} = CUDA.device_count()
        assert is_binary(reason)
    end
  end
end
