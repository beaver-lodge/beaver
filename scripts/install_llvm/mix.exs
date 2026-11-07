defmodule BeaverLlvmInstaller.MixProject do
  use Mix.Project

  def project do
    [
      app: :beaver_llvm_installer,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: false,
      deps: [
        toolchain(:mlir, "MLIR_TOOLCHAIN_PATH"),
        toolchain(:openai_triton, "OPENAI_TRITON_PATH")
      ]
    ]
  end

  defp toolchain(app, env) do
    case System.get_env(env) do
      nil -> {app, "~> 0.1.0"}
      path -> {app, path: Path.expand(path), override: true}
    end
  end

  def application do
    [extra_applications: [:crypto, :inets, :ssl]]
  end
end
