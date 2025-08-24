args = System.argv() |> Enum.chunk_every(2)

[["--mlir-include-dir", mlir_include_dir], ["--output", output]] =
  args

mlir_include_dir = Path.expand(mlir_include_dir)

files =
  ~w{mlir-c/**/*.h}
  |> Enum.flat_map(&Path.wildcard(Path.join(mlir_include_dir, &1)))
  |> Enum.reject(&String.contains?(&1, "Python"))
  |> Enum.reject(&String.contains?(&1, "LLVMIR"))
  |> Enum.map(&Path.relative_to(&1, mlir_include_dir))

if Enum.empty?(files) do
  raise "no headers found: #{mlir_include_dir}"
end

files =
  files
  |> Enum.concat(~w{
mlir-c/Beaver/Context.h
mlir-c/Beaver/Op.h
mlir-c/Beaver/Pass.h
mlir-c/Dialect/Elixir.h
})
  |> Enum.sort()

File.write!(
  output,
  files
  |> Enum.map(
    &"""
    #include <#{&1}>
    """
  )
  |> Enum.join()
)
