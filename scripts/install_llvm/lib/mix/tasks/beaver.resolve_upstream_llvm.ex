defmodule Mix.Tasks.Beaver.ResolveUpstreamLlvm do
  use Mix.Task
  @moduledoc false

  @shortdoc "Freeze the latest upstream CI LLVM build and its four platform artifacts"

  def run(args, opts \\ []) do
    {switches, positional} =
      OptionParser.parse!(args, strict: [build_run: :string, github_output: :string])

    if positional != [], do: Mix.raise("unexpected positional arguments")

    manifest =
      Beaver.ToolchainMetadata.load!()["llvm"]
      |> Beaver.UpstreamLLVM.resolve!(switches[:build_run] || "latest", opts)
      |> JSON.encode!()

    IO.puts(manifest)

    if output = switches[:github_output],
      do: File.write!(output, "llvm_ci_manifest=#{manifest}\n", [:append])
  end
end
