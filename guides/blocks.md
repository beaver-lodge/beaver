# Blocks and the underscore convention

MLIR blocks are the basic units of control flow inside a region. In Beaver's
DSL, a `block` is a lexically scoped value: you create it with the `block`
macro, reference it by name with `Beaver.Env.block/1`, and it lives inside the
enclosing `region`.

## Named and anonymous blocks

```elixir
mlir ctx: ctx do
  module do
    Func.func some_func(function_type: Type.function([], [Type.i32()])) do
      region do
        # anonymous block: the underscore says "this block is incidental"
        block _() do
          v0 = Arith.constant(value: Attribute.integer(Type.i32(), 0)) >>> Type.i32()
          CF.br({Beaver.Env.block(bb1), [v0]}) >>> []
        end

        # named block: the name is a value other terminators can reference
        block bb1(arg >>> Type.i32()) do
          Func.return(arg) >>> []
        end
      end
    end
  end
end
```

A block can be referenced **before** it is defined. `Beaver.Env.block(bb1)`
creates the block on first use, and the later `block bb1(...) do ... end`
definition appends it to the region. This is how terminators form control-flow
edges between blocks that are written out of order.

## Dangling blocks

A block is *dangling* when it was created (usually by a forward reference) but
never appended to a region. It is not reachable from the module, and any
terminator that jumps to it references invalid IR.

Beaver reuses Elixir's own conventions to surface this:

- **Plain names are tracked.** If a plain-named block is created but never
  appended, the region raises at the end of its body:

  ```text
  ** (ArgumentError) dangling blocks created but never appended to a region: [:bb2].
  Define each referenced block with `block <name>() do ... end` inside the region,
  or prefix the variable with an underscore to opt out of this check.
  ```

- **Underscore-prefixed names opt out.** `block _bb1(...) do ... end` and
  `block _() do ... end` say "this block is not meant to be part of the final
  IR", so they are exempt from the runtime check. A dangling underscore block
  still fails MLIR verification if it is referenced, but the DSL itself lets
  it through — the same way `_` suppresses unused-variable warnings in Elixir.

- **The compiler warns too.** Creating a named block without ever using its
  binding produces Elixir's ordinary unused-variable warning, pointing at the
  block that was never joined into the region.

## Three layers of detection

1. **Compile time** — Elixir reports unused block variables (`variable "bb2"
   is unused`). This is the same mechanism as unused function arguments: cheap
   and local.
2. **Region end** — the DSL verifies that every plain-named block was appended,
   and raises with the offending names before you ever build the IR.
3. **MLIR verification** — the final safety net. `MLIR.verify!/1` rejects a
   module whose terminators reference a block in another region, with the
   standard "reference to block defined in another region" diagnostic.

## When to use which

- Use a **plain name** for every block you intend to be part of the CFG, even
  if it is written after its first reference.
- Use `_` or an **underscore-prefixed name** for scratch blocks that are
  created for a side effect (for example, exploring the API) and are never
  meant to survive.
- Keep `MLIR.verify!/1` in your pipeline: it catches the cases the DSL check
  deliberately lets through, and it is the contract Triton and other backends
  rely on.
