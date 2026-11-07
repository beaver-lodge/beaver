defmodule Mix.Tasks.Beaver.InstallPrebuiltLlvm do
  use Mix.Task
  @shortdoc "Install Beaver's selected LLVM/MLIR toolchain"
  def run(args) do
    {opts, positional} = OptionParser.parse!(args, strict: MLIR.Toolchain.switches())

    if opts[:print_metadata] do
      if positional != [], do: Mix.raise("--print-metadata does not accept positional arguments")
      IO.puts(Beaver.ToolchainMetadata.encode!())
    else
      MLIR.Toolchain.run(args, metadata: Beaver.ToolchainMetadata.load!()["llvm"])
    end
  end
end
