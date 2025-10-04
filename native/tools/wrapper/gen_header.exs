{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [mlir_include_dir: :string, output: :string]
  )

mlir_include_dir = Path.expand(Keyword.fetch!(opts, :mlir_include_dir))
output = Keyword.fetch!(opts, :output)

files =
  ~w{mlir-c/**/*.h}
  |> Stream.flat_map(&Path.wildcard(Path.join(mlir_include_dir, &1)))
  |> Stream.reject(&String.contains?(&1, "mlir-c/Bindings/Python"))
  |> Stream.reject(&String.contains?(&1, "mlir-c/Target/LLVMIR.h"))
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
mlir-c/Beaver/Debug.h
mlir-c/Dialect/Elixir.h
mlir-c/Beaver/CallbackTypeDef.h
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
