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
## Reusable compiler building blocks

| API | Responsibility |
| --- | --- |
| `IRBatch.Builder` | Immutable blocks, scoped constants, deferred expressions, multi-result calls, guarded joins and dispatch. |
| `IRBatch.Spec` | Map/keyword/tuple normalization, declared type aliases and call shortcuts; custom literal expansion. |
| `Template.Library` | Revision-keyed bytecode and signatures; each use owns independent IR in the caller's context. |
| `SymbolTable` | Check complete declaration batches and transitive imports before changing the target. |
| `Pattern.PDL` | Explicit callee metadata (`:all` for unrestricted rules); reconsider rules when rewrites introduce callees. |
| `CABI.wrapper_source/4` | Value wrappers using the same conventions as `Pattern.Native.Program.c_abi_call/3`. |
| `SSA.Schema` / `SSA.BlockSharing` | Traverse declared value/edge paths; share alpha-equivalent bodies with caller eligibility/literal/jump policies. |
| `Build.Plan` / `Build.ArtifactCache` | Stage dependencies and timings; digest checks and atomic publication of artifacts before manifests. |

Use `builder.data` for language state. Sharing preserves metadata/order and excludes entry blocks and escaping definitions.
Cache revisions must include every semantic input and generator dependency. `LLVMIR.compile_units!` owns parallel object
compilation. Each rename is atomic; publication of multiple files is not a filesystem transaction.

`IRBatch.CFG` adds hygienic syntax: bind effectful expressions with `<-` before reuse; `step` runs a builder callback,
`emit` discards its result, and `dispatch` emits runtime arms. Duplicate static names fail at expansion; Builder checks
block closure and MLIR checks types/dominance. Submit `Builder.finish(builder)` through `IRBatch.build_function!/3`.

`IRBatch.Protocol` and Zig's `batch_protocol.zig` derive field positions and operand tags from `protocol.json`.
Version 1 preserves seven-field single-block and eight-field CFG tuples, with native bounds/context/resource checks.
`Protocol.native_schema(ctx)` detects NIF/schema mismatches.
