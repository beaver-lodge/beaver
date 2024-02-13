ast_macros = [op: 1, op: 2, op: 4, op: 5]
locals_without_parens = [deftype: 1, defattr: 1, defop: 2, defalias: 2] ++ ast_macros

# Used by "mix format"
[
  locals_without_parens: locals_without_parens,
  inputs: ["{mix,.formatter}.exs", "{config,lib,test}/**/*.{ex,exs}"],
  export: [
    locals_without_parens: locals_without_parens
  ]
]
