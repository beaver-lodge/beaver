defmodule Beaver.MixProject do
  use Mix.Project

  def project do
    [
      app: :beaver,
      version: "0.1.0",
      build_path: "../../_build",
      config_path: "../../config/config.exs",
      deps_path: "../../deps",
      lockfile: "../../mix.lock",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package()
    ]
  end

  defp description() do
    "Beaver, a MLIR Toolkit in Elixir"
  end

  defp package() do
    [
      licenses: ["Apache-2.0", "MIT"],
      links: %{"GitHub" => "https://github.com/beaver-project/beaver"}
    ]
  end

  def application do
    [
      mod: {Beaver.Application, []},
      start_phases: [load_c_library: [], generate_c_apis: [], generate_dialects: []],
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:exotic, in_umbrella: true},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end
end
