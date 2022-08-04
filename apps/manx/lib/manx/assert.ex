defmodule Manx.Assert do
  @moduledoc """
  Tensor assertions. Original implementation is from EXLA.
  """

  defmacro assert_equal(left, right) do
    # Assert against binary backend tensors to show diff on failure
    quote do
      assert unquote(left) |> Manx.Assert.to_binary_backend() ==
               unquote(right) |> Manx.Assert.to_binary_backend()
    end
  end

  def to_binary_backend(tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  def all_close(left, right, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-4)
    rtol = Keyword.get(opts, :rtol, 1.0e-4)

    equals =
      left
      |> Nx.all_close(right, atol: atol, rtol: rtol)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if equals != Nx.tensor(1, type: {:u, 8}, backend: Nx.BinaryBackend) do
      raise("""
      expected

      #{inspect(left)}

      to be within tolerance of

      #{inspect(right)}
      """)
    end
  end

  defmacro assert_all_close(left, right, opts \\ []) do
    quote bind_quoted: [
            left: left,
            right: right,
            opts: opts
          ] do
      all_close(left, right, opts)
    end
  end
end
