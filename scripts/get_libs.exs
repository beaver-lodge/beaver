parts =
  "_build/#{System.get_env("MIX_ENV")}/lib/beaver/cmake_build/build.ninja"
  |> File.read!()
  |> String.split()

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
  |> Enum.uniq()
  |> Enum.sort()
  |> Enum.filter(& &1)

txt = libs |> Enum.map(&("\"" <> &1 <> "\"")) |> Enum.join(", ")

txt = """
pub const mlirLibs = .{ #{txt} };
"""

File.write!("native/mlir-zig-proj/libs.zig", txt)
