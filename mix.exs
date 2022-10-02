defmodule Beaver.Umbrella.MixProject do
  use Mix.Project

  def project do
    [
      apps_path: "apps",
      version: "0.1.0",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "MLIR",
      source_url: "https://github.com/beaver-project/beaver",
      homepage_url: "https://hexdocs.pm/beaver",
    ]
  end

  defp deps do
    [{:ex_doc, "~> 0.27", only: :dev, runtime: false}]
  end
end
