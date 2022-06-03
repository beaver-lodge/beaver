defmodule Met.MixProject do
  use Mix.Project

  def project do
    [
      apps_path: "apps",
      version: "0.1.0",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      # Docs
      name: "MLIR",
      source_url: "https://github.com/USER/PROJECT",
      homepage_url: "http://YOUR_PROJECT_HOMEPAGE",
      docs: [
        # The main page in the docs
        # main: "MLIR",
        extras: ["README.md"]
      ]
    ]
  end

  # Dependencies listed here are available only for this
  # project and cannot be accessed from applications inside
  # the apps folder.
  #
  # Run "mix help deps" for examples and options.
  defp deps do
    [{:ex_doc, "~> 0.27", only: :dev, runtime: false}]
  end
end
