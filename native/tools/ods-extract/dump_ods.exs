{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [mlir_include_dir: :string, output: :string]
  )

mlir_include_dir = Path.expand(Keyword.fetch!(opts, :mlir_include_dir))
output = Keyword.fetch!(opts, :output)

files =
  ~w{mlir/Dialect/**/*.td}
  |> Stream.flat_map(&Path.wildcard(Path.join(mlir_include_dir, &1)))
  |> Stream.reject(&String.contains?(&1, "Enum"))
  |> Stream.reject(&String.contains?(&1, "Attr"))
  |> Stream.reject(&String.contains?(&1, "Pass"))
  |> Stream.reject(&String.contains?(&1, "Base"))
  |> Stream.reject(&String.contains?(&1, "Type"))
  |> Stream.reject(&String.contains?(&1, "Transform"))
  |> Stream.reject(&String.contains?(&1, "yamlgen"))
  |> Stream.reject(&String.contains?(&1, "Dialect.td"))
  |> Stream.reject(&String.contains?(&1, "OpenACC"))
  |> Stream.reject(&String.contains?(&1, "OpenMP"))
  |> Stream.reject(&String.contains?(&1, "DialectBytecode.td"))
  |> Stream.reject(&String.contains?(&1, "SMT"))
  |> Stream.reject(&String.contains?(&1, "SPIRV"))
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
