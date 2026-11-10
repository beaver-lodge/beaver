# Passes and patterns

`Beaver.Composer` assembles pass pipelines. A pass can use
`Beaver.Pattern.defpat` to describe a rewrite as PDL IR, then apply it to a
module with `Beaver.MLIR.apply_/2`.

## Define a PDL-backed pass

This example replaces `tosa.add` with `tosa.sub`. It demonstrates the rewrite
mechanics, not a semantics-preserving optimization.

<!-- beaver:test:pass-definition -->
```elixir
defmodule ToyPass do
  @moduledoc false
  use Beaver
  alias MLIR.Dialect.TOSA
  import Beaver.Pattern
  use MLIR.Pass, on: "builtin.module"

  defpat replace_add_op() do
    a = value()
    b = value()
    res = type()
    {op, _t} = TOSA.add(a, b) >>> {:op, [res]}

    rewrite op do
      {r, _} = TOSA.sub(a, b) >>> {:op, [res]}
      replace(op, with: r)
    end
  end

  def run(%MLIR.Operation{} = operation, _state) do
    with 1 <- Beaver.Walker.regions(operation) |> Enum.count(),
         {:ok, _} <-
           MLIR.apply_(MLIR.Module.from_operation(operation), [replace_add_op(benefit: 2)]) do
      :ok
    else
      _ -> raise "rewrite failed"
    end
  end
end
```

## Run the pipeline

Put the definition above and the following code in one `.exs` file in a project
with Beaver installed, then run it with `mix run`. It parses a module, applies
the custom pass and canonicalization, verifies the result, and prints the
rewritten IR. Both the module and its context are explicitly released.

<!-- beaver:test:pass-pipeline -->
```elixir
use Beaver
import MLIR.Transform

ctx = MLIR.Context.create()

try do
  ir =
    ~m"""
    module {
      func.func @tosa_add(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
        %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
        return %0 : tensor<2x3xf32>
      }
    }
    """
    |> Beaver.Deferred.resolve(ctx)

  try do
    ir
    |> Beaver.Composer.append(ToyPass)
    |> canonicalize()
    |> Beaver.Composer.run!()
    |> MLIR.verify!()
    |> MLIR.to_string()
    |> IO.puts()
  after
    MLIR.Module.destroy(ir)
  end
after
  MLIR.Context.destroy(ctx)
end
```

The printed function contains `tosa.sub` instead of `tosa.add`. For rewrites
that need arbitrary Elixir checks or callback state, see
[native rewrite patterns](native-rewrite-patterns.md). For coordinated type
and operation lowering, see [dialect conversion](dialect-conversion.md).
