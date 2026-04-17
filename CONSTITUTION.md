# Beaver Constitution

This repository is an MLIR/LLVM toolkit for Elixir.

## Closure

- Primary closure is `mix test`.
- Build and test runs depend on:
  - `LLVM_CONFIG_PATH`
  - a working Zig toolchain
- Prefer fixes that preserve the current Elixir API and DSL shape.

## Mutation Discipline

- Prefer the smallest verifier-relevant diff.
- Do not rewrite generated or low-level binding surfaces unless the verifier
  failure is clearly rooted there.
- Treat `build.zig`, `native/`, and MLIR CAPI-facing code as stricter surfaces
  than ordinary Elixir modules.

## Scope

- Default scope is the target file plus a bounded local neighborhood.
- Widen only when compile/test failures show that the current scope is
  insufficient.
- If a fix must cross into sibling repos or dependency internals, surface that
  explicitly instead of smuggling it through unrelated edits.

## Toolchain

- LLVM is currently expected from `LLVM_CONFIG_PATH`.
- On this machine the active LLVM install is `/Users/tsai/Downloads/llvm-install`.
- Prefer `zig@0.15` for now. Zig 0.16 compatibility is a separate migration
  surface.
