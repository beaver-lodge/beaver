# Callback rewrite patterns

Register a four-argument callback with an explicit operation root:

```elixir
def replace(_pattern, operation, rewriter, state) do
  case Enum.to_list(Beaver.Walker.operands(operation)) do
    [value] ->
      Beaver.MLIR.PatternRewriter.replace_op(rewriter, operation, value)
      {:ok, state}
    _ ->
      {:error, state}
  end
end

Beaver.MLIR.Rewrite.apply_patterns!(ir, [{"example.identity", &replace/4}])
```

A successful callback must mutate IR through `PatternRewriter` or `RewriterBase`.
Return `{:ok, new_state}` for a match and `{:error, new_state}` for no match.
`RewritePatternSet.add/4` additionally accepts benefit and lifecycle options.
Pattern processes serialize state transitions; callback handles are borrowed
and must not be retained or sent to another process.

For declarative PDL, construct `MLIR.Dialect.PDL` operations directly and use
`RewritePatternSet.with_pdl_patterns/2`. The old `Beaver.Pattern` and
`Beaver.Pattern.Native` DSLs have been retired; both underlying rewrite paths
remain available.
