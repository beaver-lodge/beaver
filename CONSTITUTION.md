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
- The documented local prefix is `priv/llvm-prebuilt`, populated from the
  `llvm/eudsl` revision pinned in `native-deps.json` by the
  `beaver.install_prebuilt_llvm` Mix task
  (`scripts/install_llvm` delegates to the standalone `mlir` and
  `openai_triton` packages; the shell scripts are compatibility wrappers).
- The active Zig line is `0.16`.
