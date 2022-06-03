# Beaver WASM compiler

## Different approaches

### Ship with a BEAM runtime

- BEAM runtime runs .beam

  - maximum compatibility
  - less efficient

- BEAM runtime runs .wasm
  - learn from the asmjit implementation in erts

### No runtime

- Doesn't support advanced feature.

## Adaptor for scheduler threading

- The threading for scheduler should be adaptive:
  - GCD on Apple platforms
  - Web worker on WASM

## Used in Livebook

- To support Livebooks, the whole compiler and runtime should be included.

## Typed vs Opaque Erlang terms

- If a function's type is declared and checked, generate a typed function for better performance.
