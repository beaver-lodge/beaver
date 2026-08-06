defmodule BeaverLlvmInstaller.MixProject do
  use Mix.Project

  def project do
    [
      app: :beaver_llvm_installer,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: false,
      deps: []
    ]
  end

  def application do
    [extra_applications: [:crypto, :inets, :ssl]]
  end
end
