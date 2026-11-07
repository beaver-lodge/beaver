{opts, _, _} =
  OptionParser.parse(System.argv(), strict: [mlir_include_dir: :string, beaver_include_dir: :string, output: :string])

mlir_include_dir = opts |> Keyword.fetch!(:mlir_include_dir) |> Path.expand()
beaver_include_dir = opts |> Keyword.fetch!(:beaver_include_dir) |> Path.expand()
output = opts |> Keyword.fetch!(:output) |> Path.expand()

mlir_headers =
  "mlir-c/**/*.h"
  |> then(&Path.join(mlir_include_dir, &1))
  |> Path.wildcard()
  |> Stream.reject(&String.contains?(&1, "mlir-c/Bindings/Python"))
  # LLVMIR and Remarks use Beaver adapters; callback-scoped remarks never escape to BEAM.
  |> Stream.reject(&String.ends_with?(&1, ~w[mlir-c/Target/LLVMIR.h mlir-c/Remarks.h]))
  |> Enum.map(&Path.relative_to(&1, mlir_include_dir))

if mlir_headers == [], do: raise("no MLIR C headers found under #{mlir_include_dir}")

beaver_headers = ~w[
  mlir-c/Beaver/CAPIPolicy.h mlir-c/Beaver/Context.h mlir-c/Beaver/CompilerKernel.h
  mlir-c/Beaver/Interfaces.h mlir-c/Beaver/Op.h mlir-c/Beaver/Pass.h mlir-c/Beaver/Debug.h mlir-c/Beaver/LLVMIR.h
  mlir-c/Beaver/Analysis.h mlir-c/Beaver/CallbackTypeDef.h mlir-c/Beaver/Remarks.h mlir-c/Beaver/ActionTracing.h
]

Enum.each(beaver_headers, fn header ->
  unless File.regular?(Path.join(beaver_include_dir, header)), do: raise("missing Beaver C API header: #{header}")
end)

body =
  mlir_headers |> Kernel.++(beaver_headers) |> Enum.uniq() |> Enum.sort() |> Enum.map_join("", &"#include <#{&1}>\n")

File.mkdir_p!(Path.dirname(output))
File.write!(output, "#pragma once\n\n" <> body)
