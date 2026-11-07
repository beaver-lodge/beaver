defmodule Mix.Tasks.Beaver.BuildTriton do
  use Mix.Task
  @shortdoc "Build Triton against the current LLVM API"
  @patch Path.expand("../../../patches/triton-llvm-main.patch", __DIR__)
  def run(args) do
    {opts, []} =
      OptionParser.parse!(args,
        strict: [source: :string, llvm: :string, output: :string, ccache: :string, cache: :string]
      )

    source =
      opts[:source] || System.get_env("TRITON_SOURCE_DIR") || Mix.raise("Triton source required")

    # A narrow API migration for the pinned Triton source, applied idempotently.
    case System.cmd("git", ["-C", source, "apply", "--reverse", "--check", @patch],
           stderr_to_stdout: true
         ) do
      {_, 0} -> :ok
      _ -> Beaver.LLVM.run!("git", ["-C", source, "apply", @patch])
    end

    output = OpenAITriton.Builder.build!(opts)
    record_identity!(output, source, opts[:llvm] || System.fetch_env!("LLVM_SYSPATH"))
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
      JSON.encode!(
        Map.merge(metadata, %{"llvm_ref" => llvm_ref, "triton_ref" => String.trim(triton_ref)})
      )
    )
  end
end
