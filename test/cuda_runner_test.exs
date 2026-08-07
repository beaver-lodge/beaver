defmodule CudaRunnerTest do
  use Beaver.Case, async: true
  alias Beaver.MLIR.CUDA

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
