parts =
  "_build/test/lib/beaver/cmake_build/build.ninja"
  |> File.read!()
  |> String.split()

libs =
  for p <- parts do
    if String.ends_with?(p, ".a") and not String.contains?(p, "Elixir") do
      "lib" <> lib = Path.basename(p)
      lib |> Path.basename(".a")
    end
  end
  |> Enum.uniq()
  |> Enum.sort()
  |> Enum.filter(& &1)

libs |> Enum.map(&("\"" <> &1 <> "\"")) |> Enum.join(", ") |> IO.puts()
