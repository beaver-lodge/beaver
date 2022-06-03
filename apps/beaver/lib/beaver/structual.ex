defmodule Beaver.Structual do
  @moduledoc """
  `Beaver.Structual` mainly gets the inspirations from [taichi](https://github.com/taichi-dev/taichi.git) and Elixir [Structs](https://elixir-lang.org/getting-started/structs.html). Here are the core features and characteristics:
  - Enumerable at all level, you can use Elixir's `for` comprehension to iterate over the fields at different level of the data stucture hierarchy.
  - Mutable, unlike most things in Erlang/Elixir. This is designed to be used in computation so being mutable is a must.
  - Automatic parallelization, just like kernels in taichi. Beaver's MLIR based compiler will compile the code to a parallelized version untilizing SIMD and GPU.
  - Adaptive structural typing. `Structual.Function` should accept any data with the fields compatible with the function's signature and different flavors of the kernel optimized for different sparsity will be automatically generated and selected.

  Major differences from taichi:
  - No template APIs for kernels.
  - It is possible to provide different kernel implementation by defining multi-cluase functions. For instance:
  ```elixir
  defk sum(a, b) when is_aos(a), do: impl1
  defk sum(a, b) when is_soa(b), do: impl2
  ```

  Major differences from Elixir:
  - Scoped. You may use `Beaver.Structual.scope/1` to define a scope in the kernel.
  """

  defmodule Function do
    @moduledoc """
    Although module is called `Function`, it is actually "kernel". We have to use an alternative name because `Kernel` is a keyword in Elixir.
    """
  end
end
