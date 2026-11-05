locals_without_parens = [
  deftype: 1,
  deftype: 2,
  defattr: 1,
  defattr: 2,
  defop: 1,
  defop: 2,
  defalias: 2,
  defconstraint: 2,
  defrewrite: 2,
  defrewrite: 3,
  defschedule: 1,
  sequence: 0,
  sequence: 1,
  sequence: 2,
  alternatives: 1
]

# Used by "mix format"
[
  locals_without_parens: locals_without_parens,
  inputs: ["{mix,.formatter}.exs", "{config,test,scripts,bench,profile}/**/*.{ex,exs}"],
  subdirectories: ["lib", "native"],
  export: [
    locals_without_parens: locals_without_parens
  ]
]
