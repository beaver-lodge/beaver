if Version.match?(System.version(), "< 1.18.0") do
  Mix.install([
    {:jason, "~> 1.4"}
  ])
end

args = System.argv() |> Enum.chunk_every(2)
[["--input", input], ["--output", output]] = args

json_mod = if Version.match?(System.version(), "< 1.18.0"), do: Jason, else: JSON

txt =
  input
  |> File.read!()
  |> String.trim()

content =
  apply(json_mod, :decode!, [txt])
  |> Enum.sort()
  |> inspect(pretty: true, limit: :infinity)

File.write!("#{output}", content)
