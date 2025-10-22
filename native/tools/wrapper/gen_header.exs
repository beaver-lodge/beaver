{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [mlir_include_dir: :string, output: :string]
  )

mlir_include_dir = Path.expand(Keyword.fetch!(opts, :mlir_include_dir))
output = Keyword.fetch!(opts, :output)

mlir_headers =
  ~w{mlir-c/**/*.h}
  |> Stream.flat_map(&Path.wildcard(Path.join(mlir_include_dir, &1)))
  |> Stream.reject(&String.contains?(&1, "mlir-c/Bindings/Python"))
  |> Stream.reject(&String.contains?(&1, "mlir-c/Target/LLVMIR.h"))
  |> Enum.map(&Path.relative_to(&1, mlir_include_dir))

if Enum.empty?(mlir_headers) do
  raise "no headers found: #{mlir_include_dir}"
end

beaver_headers = ~w{
mlir-c/Beaver/Context.h
mlir-c/Beaver/Op.h
mlir-c/Beaver/Pass.h
mlir-c/Beaver/Debug.h
mlir-c/Dialect/Elixir.h
mlir-c/Beaver/CallbackTypeDef.h
}

# Create output directory for JSON files
json_output_dir = Path.join(Path.dirname(output), "extract-api")
File.mkdir_p!(json_output_dir)

# Process files concurrently using Task.async_stream
zig = System.find_executable("zig") || raise "zig executable not found"

stream =
  Task.async_stream(
    mlir_headers,
    fn f ->
      full_path = Path.join(mlir_include_dir, f)
      # Generate JSON API extraction for each header
      basename = Path.basename(f, ".h")
      json_output = Path.join(json_output_dir, "#{basename}.json")

      cmd_args = [
        "cc",
        "-E",
        "-Xclang",
        "-extract-api",
        "-x",
        "c-header",
        full_path,
        "-I",
        "include",
        "-I",
        mlir_include_dir,
        "-o",
        json_output
      ]

      {cmd_output, exit_code} = System.cmd(zig, cmd_args)

      if exit_code != 0 do
        IO.puts(:stderr, "Warning: Failed to extract API for #{f}: #{cmd_output}")
      end

      {f, exit_code}
    end,
    max_concurrency: System.schedulers_online(),
    ordered: false
  )

# Run the stream to execute all tasks
Stream.run(stream)

files = mlir_headers |> Enum.concat(beaver_headers) |> Enum.sort()

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
