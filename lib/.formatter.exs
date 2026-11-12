{opts, _} = Code.eval_file(Path.expand("../.formatter.exs", __DIR__))

[inputs: ["**/*.{ex,exs}"], line_length: 120, locals_without_parens: opts[:locals_without_parens]]
