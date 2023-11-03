locals_without_parens = [deftype: 1, deftype: 2, defop: 1, defop: 2, defalias: 1, defalias: 2]
# Used by "mix format"
[
  inputs: ["{mix,.formatter}.exs", "{config,lib,test}/**/*.{ex,exs}"],
  export: [
    locals_without_parens: locals_without_parens
  ]
]
