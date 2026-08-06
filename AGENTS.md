# Beaver Agent Guidelines

Rules for AI agents working in this repository.

## Elixir Tests

- All tests must support parallel execution (`use ExUnit.Case, async: true`).
- Avoid test designs that mutate process-global or VM-global state such as cwd
  or OS env; prefer explicit options, injected dependencies, and
  `@tag :tmp_dir`.
- Do not use the Elixir/Erlang process dictionary in repo code or tests.
- Use ExUnit's built-in `@tag :tmp_dir` and the test context
  `%{tmp_dir: tmp_dir}` instead of manual `System.tmp_dir!()` and cleanup.

Known exception: `test/expandable_jit/native_heap_global_test.exs` is the only
sanctioned `async: false` module. It covers the VM-global heap ABI (global GC
mark bits, dangling-pointer verification), whose semantics cannot hold under
concurrent execution. New global-heap coverage must be added to that file
rather than introducing another `async: false` module.

## Elixir JSON

- Use Elixir's current built-in `JSON` module for JSON encoding and decoding.
- Do not add, reintroduce, or fallback to `Jason`; existing Jason call sites
  must be migrated to the built-in module.
- Beaver requires Elixir >= 1.18 (the first release with built-in JSON) and
  declares that floor in `mix.exs` (`elixir: "~> 1.18"`).
- Prefer direct calls such as `JSON.decode!/1` and `JSON.encode!/1` in repo
  code and tests.

## Command Orchestration

- Prefer structured Elixir Mix tasks to wrap command invocation instead of
  adding shell scripts.
- Shell scripts are acceptable only as thin compatibility wrappers or when the
  command boundary is inherently shell-native.
