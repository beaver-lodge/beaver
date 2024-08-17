locals_without_parens = [deftype: 1, defattr: 1, defop: 2, defalias: 2]
# Used by "mix format"
[
  locals_without_parens: locals_without_parens,
  inputs: ["{mix,.formatter}.exs", "{config,lib,test,scripts,bench,profile,native}/**/*.{ex,exs}"],
  export: [
    locals_without_parens: locals_without_parens
  ]
]
