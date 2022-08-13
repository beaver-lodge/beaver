defmodule Manx.VulkanExprTest do
  # TODO: running this in async will trigger multi-thread check in MLIR and crash
  use ExUnit.Case, async: true
  # import Nx, only: :sigils
  import Nx.Defn
  # import Manx.Assert

  @moduletag :nx
  @moduletag :vulkan
  setup do
    Nx.Defn.default_options(compiler: Manx.Compiler)
    :ok
  end

  describe "unary float ops" do
    @float_tensor Nx.tensor([1.0, 2.0, 3.0])
    defn unary_sin(t), do: Nx.sin(t)

    test "sin" do
      # assert [0.8414709568023682, 0.9092974066734314, 0.14112000167369843] ==
      for _i <- 0..1 do
        r = unary_sin(@float_tensor)
        r |> Nx.to_flat_list() |> IO.inspect()
      end
    end
  end
end
