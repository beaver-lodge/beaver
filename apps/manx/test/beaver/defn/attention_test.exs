defmodule Manx.AttentionTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  @moduletag :nx
  @moduletag :attention
  setup do
    Nx.Defn.default_options(compiler: Manx.Compiler)
    :ok
  end

  # original implementation from: https://github.com/sooftware/attentions/blob/master/attentions.py
  describe "attention" do
    defn(softmax(t), do: Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [-1], keep_axes: true))

    defn(batched_dot(t1, t2), do: Nx.dot(t1, [1], [0], t2, [1], [0]))

    @doc """
    dim is the dimension of each head
    """
    defn scaled_dot_product_attention(dim, query, key, value) do
      score = Nx.dot(query, [2], [0], key, [2], [0]) / Nx.sqrt(dim)
      attn = softmax(score)
      batched_dot(attn, value)
    end

    test "dot product attention" do
      # do a divide to prevent overflow
      query = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)
      key = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)
      value = Nx.iota({4, 3, 2}, type: {:f, 32}) |> Nx.divide(10.0)
      scaled_dot_product_attention(1, query, key, value)
    end
  end
end
