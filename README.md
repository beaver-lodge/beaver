# Beaver 🦫

[![Package](https://img.shields.io/badge/-Package-important)](https://hex.pm/packages/beaver) [![Documentation](https://img.shields.io/badge/-Documentation-blueviolet)](https://hexdocs.pm/beaver)
[![Check Upstream](https://github.com/beaver-lodge/beaver/actions/workflows/upstream.yml/badge.svg)](https://github.com/beaver-lodge/beaver/actions/workflows/upstream.yml)
[![Run in Livebook](https://livebook.dev/badge/v1/black.svg)](https://livebook.dev/run?url=https%3A%2F%2Fhexdocs.pm%2Fbeaver%2Fyour-first-beaver-compiler.livemd)

Beaver is an MLIR toolkit for Elixir. It combines an SSA-oriented DSL with
bindings to MLIR so you can build, transform, and execute IR from the BEAM.
It supplies compiler building blocks, not a complete Elixir compiler or an
application runtime.

- Construct and inspect IR with Elixir macros and collection APIs.
- Define passes, rewrite patterns, and dynamic dialects; compose lowering pipelines.
- Use MLIR execution engines and the Triton/CUDA APIs for native and GPU work.

## Installation

Requires Elixir 1.18 or later. Add Beaver to your project's `mix.exs`:

```elixir
def deps do
  [
    {:beaver, "~> 0.4.0"}
  ]
end
```

Run `mix deps.get`, and add `import_deps: [:beaver]` to the keyword list in
`.formatter.exs` to enable formatting for Beaver's DSL.

If a native build is needed, use the LLVM/MLIR revision pinned in
`native-deps.json`, or a newer revision with the same APIs. Toolchain installation,
Triton builds, and contributor setup are covered in the
[contributing guide](CONTRIBUTING.md#development).

## Build and verify IR

Save this complete example as `example.exs` in your project and run
`mix run example.exs`. It prints a verified MLIR module containing an `i32`
constant; it does not compile or execute a function.

<!-- beaver:test:quickstart -->
```elixir
use Beaver
alias Beaver.MLIR.Dialect.Arith

ctx = MLIR.Context.create()

try do
  ir =
    mlir ctx: ctx do
      module do
        Arith.constant(value: ~a{42 : i32}) >>> ~t{i32}
      end
    end

  try do
    ir |> MLIR.verify!() |> MLIR.to_string() |> IO.puts()
  after
    MLIR.Module.destroy(ir)
  end
after
  MLIR.Context.destroy(ctx)
end
```

The `>>>` operator declares an operation's result types. The enclosing `mlir`
and `module` macros supply its context and insertion point. Builders without
`ctx:` return explicit `Beaver.Deferred` values; there is no process-global
MLIR context. See [context ownership](guides/architecture.md#context-ownership).

## Guides

- **Start here:** [Your first compiler](guides/your-first-beaver-compiler.livemd)
  walks through IR construction, lowering, and JIT execution.
- **Build IR:** [Blocks and control flow](guides/blocks.md),
  [dynamic dialects with Slang](guides/slang.md).
- **Transform IR:** [Passes and patterns](guides/passes-and-patterns.md),
  [native rewrite callbacks](guides/native-rewrite-patterns.md),
  [dialect conversion](guides/dialect-conversion.md).
- **Execute and inspect:** [Direct compilation](guides/incremental-compilation-runtime.md),
  [Transform execution](guides/transform-autotuning.md),
  [action tracing](guides/action-tracing.md),
  [conversion profiling](guides/conversion-profiling.md).
- **Understand the implementation:** [Architecture](guides/architecture.md),
  [core API migration](guides/core-api-migration.md),
  [API reference](https://hexdocs.pm/beaver).

## Related projects

[Kinda](https://github.com/beaver-lodge/kinda) supplies the Zig/Elixir binding framework used by Beaver. [Sluice](https://github.com/batata-lang/sluice) builds a Batata compiler provider on Beaver. The [Batata Compiler organization](https://github.com/batata-lang) groups frontend, compiler, ABI and runtime work; Beaver remains a general MLIR toolkit.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development, testing, and release
instructions. Documentation examples are tested alongside the code.
