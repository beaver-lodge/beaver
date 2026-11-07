defmodule Mix.Tasks.Beaver.InstallPrebuiltLlvm do
  use Mix.Task
  @shortdoc "Install Beaver's selected LLVM/MLIR toolchain"
  def run(args) do
    {opts, positional} =
      OptionParser.parse!(args,
        strict: MLIR.Toolchain.switches() ++ [from_source: :boolean, jobs: :integer]
      )

    if opts[:print_metadata] do
      if positional != [], do: Mix.raise("--print-metadata does not accept positional arguments")
      IO.puts(Beaver.ToolchainMetadata.encode!())
    else
      metadata = Beaver.ToolchainMetadata.load!()["llvm"]
      overrides = [:asset_revision, :asset_name, :asset_url]

      explicit =
        Enum.any?(overrides, &opts[&1]) or
          Enum.any?(
            ~w(LLVM_EUDSL_ASSET_REVISION LLVM_EUDSL_ASSET_NAME LLVM_EUDSL_ASSET_URL),
            &(System.get_env(&1) not in [nil, ""])
          )

      if explicit do
        MLIR.Toolchain.run(args, metadata: metadata)
      else
        opts = Enum.reject(opts, fn {_, value} -> value == "" end)

        opts =
          if positional == [], do: opts, else: Keyword.put_new(opts, :install_dir, hd(positional))

        opts =
          Enum.reduce(
            [asset_os: "LLVM_EUDSL_ASSET_OS", asset_arch: "LLVM_EUDSL_ASSET_ARCH"],
            opts,
            fn {key, env}, opts ->
              if value = System.get_env(env), do: Keyword.put_new(opts, key, value), else: opts
            end
          )

        if opts[:resolve_only] || System.get_env("LLVM_EUDSL_RESOLVE_ONLY") == "1" do
          asset = Beaver.LLVM.asset(metadata, opts)

          IO.puts(
            "LLVM_PREBUILT_ASSET_NAME=#{asset.name}\nLLVM_PREBUILT_URL=https://api.github.com/#{asset.endpoint}\nLLVM_SOURCE_REF=#{metadata["source_ref"]}"
          )
        else
          Beaver.LLVM.install!(metadata, opts)
        end
      end
    end
  end
end
