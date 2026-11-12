defmodule Mix.Tasks.Beaver.BuildTriton do
  use Mix.Task
  @shortdoc "Build Triton against the current LLVM API"
  @patches Path.expand("../../../patches", __DIR__)
  def run(args) do
    {opts, []} =
      OptionParser.parse!(args,
        strict: [source: :string, llvm: :string, output: :string, ccache: :string, cache: :string]
      )

    source =
      opts[:source] || System.get_env("TRITON_SOURCE_DIR") || Mix.raise("Triton source required")

    llvm = opts[:llvm] || System.fetch_env!("LLVM_SYSPATH")

    sparse_analysis =
      File.read!(Path.join(llvm, "include/mlir/Analysis/DataFlow/SparseAnalysis.h"))

    # LLVM main added a lattice-read callback after our pinned LLVM revision.
    for {name, needed?} <- [
          {"triton-llvm-main.patch", true},
          {"triton-dataflow-lattice.patch", String.contains?(sparse_analysis, "using GetLatticeFn =")}
        ] do
      patch = Path.join(@patches, name)

      {_, status} =
        System.cmd("git", ["-C", source, "apply", "--reverse", "--check", patch], stderr_to_stdout: true)

      if needed? != (status == 0) do
        args = if needed?, do: [patch], else: ["--reverse", patch]
        Beaver.LLVM.run!("git", ["-C", source, "apply" | args])
      end
    end

    output = OpenAITriton.Builder.build!(opts)
    record_identity!(output, source, llvm)
  end

  @doc false
  def record_identity!(output, source, llvm) do
    header = File.read!(Path.join(llvm, "include/llvm/Support/VCSRevision.h"))
    [_, llvm_ref] = Regex.run(~r/LLVM_REVISION.*?([0-9a-f]{40})/, header)
    {triton_ref, 0} = System.cmd("git", ["-C", source, "rev-parse", "HEAD"])
    path = Path.join(output, "toolchain.json")
    metadata = path |> File.read!() |> JSON.decode!()

    File.write!(
      path,
      JSON.encode!(Map.merge(metadata, %{"llvm_ref" => llvm_ref, "triton_ref" => String.trim(triton_ref)}))
    )
  end
end
