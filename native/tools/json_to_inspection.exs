{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [input: :string, output: :string]
  )

input = Path.expand(Keyword.fetch!(opts, :input))
output = Path.expand(Keyword.fetch!(opts, :output))

if Version.match?(System.version(), "< 1.18.0") do
  Mix.install([
    {:jason, "~> 1.4"}
  ])
end

decoded =
  if(Version.match?(System.version(), "< 1.18.0"), do: Jason, else: JSON)
  |> apply(:decode!, [File.read!(input)])

File.write(output, inspect(decoded, pretty: true, limit: :infinity, printable_limit: :infinity))
