txt =
  "_build/#{System.get_env("MIX_ENV")}/lib/beaver/cmake_build/build.ninja"
  |> File.read!()

lines = txt |> String.split("\n")
parts = txt |> String.split()

libs =
  for p <- parts do
    if String.ends_with?(p, ".a") and not String.contains?(p, "Elixir") do
      "lib" <> lib = Path.basename(p)
      lib |> Path.basename(".a")
    else
      if String.ends_with?(p, ".o") and not String.contains?(p, "Elixir") do
        "obj." <> lib = p |> Path.dirname() |> Path.basename()
        lib
      end
    end
  end
  |> Enum.filter(& &1)
  |> Enum.uniq()
  |> Enum.sort()

flags =
  for l <- lines do
    case l |> String.split() do
      ["FLAGS", "=" | tail] ->
        tail
        |> Enum.reject(&(&1 in ~w{-fno-lifetime-dse}))
        |> Enum.reject(&String.contains?(&1, "-std="))

      _ ->
        nil
    end
  end
  |> Enum.filter(& &1)
  |> List.first()

txt = libs |> Enum.map(&("\"" <> &1 <> "\"")) |> Enum.join(", ")
flags_txt = flags |> Enum.map(&("\"" <> &1 <> "\"")) |> Enum.join(", ")

txt = """
pub const mlirLibs = .{ #{txt} };
pub const flags = .{ #{flags_txt} };
"""

File.write!("native/mlir-zig-proj/libs.zig", txt)
