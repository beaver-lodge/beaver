defmodule Manx.VulkanExprTest do
  # TODO: running this in async will trigger multi-thread check in MLIR and crash
  use ExUnit.Case, async: true
  # import Nx, only: :sigils
  import Nx.Defn
  # import Manx.Assert

  @moduletag :nx
  @moduletag :vulkan
  setup do
    Nx.Defn.default_options(compiler: Manx.Compiler, default_backends: {Manx, device: :vulkan})
    :ok
  end

  describe "unary float ops" do
    @float_tensor Nx.tensor([1.0, 2.0, 3.0])
    defn unary_sin(t) do
      Nx.sin(t)
    end

    test "sin" do
      # assert [0.8414709568023682, 0.9092974066734314, 0.14112000167369843] ==
      r = Nx.backend_transfer(@float_tensor, {Manx, device: :vulkan}) |> unary_sin()
      r |> Nx.to_flat_list() |> IO.inspect()
    end

    @int_tensor_a Nx.tensor([1, 2, 3], type: {:u, 32})
    @int_tensor_b Nx.tensor([4, 5, 6], type: {:u, 32})
    defn binary_add(a, b) do
      Nx.add(a, b)
    end

    test "add" do
      a = Nx.backend_transfer(@int_tensor_a, {Manx, device: :vulkan})
      b = Nx.backend_transfer(@int_tensor_b, {Manx, device: :vulkan})
      binary_add(a, b)
    end
  end
end
