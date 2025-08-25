{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [mlir_include_dir: :string, output: :string]
  )

mlir_include_dir = Path.expand(Keyword.fetch!(opts, :mlir_include_dir))
output = Keyword.fetch!(opts, :output)

files =
  ~w{mlir/Dialect/**/*.td}
  |> Enum.flat_map(&Path.wildcard(Path.join(mlir_include_dir, &1)))
  |> Enum.reject(&String.contains?(&1, "Enum"))
  |> Enum.reject(&String.contains?(&1, "Attr"))
  |> Enum.reject(&String.contains?(&1, "Pass"))
  |> Enum.reject(&String.contains?(&1, "Base"))
  |> Enum.reject(&String.contains?(&1, "Type"))
  |> Enum.reject(&String.contains?(&1, "Transform"))
  |> Enum.reject(&String.contains?(&1, "yamlgen"))
  |> Enum.reject(&String.contains?(&1, "Dialect.td"))
  |> Enum.reject(&String.contains?(&1, "OpenACC"))
  |> Enum.reject(&String.contains?(&1, "OpenMP"))
  |> Enum.reject(&String.contains?(&1, "DialectBytecode.td"))
  |> Enum.reject(&String.contains?(&1, "SMT"))
  |> Enum.reject(&String.contains?(&1, "SPIRV"))
  |> Enum.map(&Path.relative_to(&1, mlir_include_dir))

File.write!(
  "#{output}.pdll",
  files
  |> Enum.map(
    &"""
    #include "#{&1}"
    """
  )
  |> Enum.join()
)

File.write!(
  "#{output}.td",
  files
  |> Enum.map(
    &"""
    include "#{&1}"
    """
  )
  |> Enum.join()
)

File.write!(
  "#{output}.txt",
  files
  |> Enum.map(
    &"""
    #{&1}
    """
  )
  |> Enum.join()
)
