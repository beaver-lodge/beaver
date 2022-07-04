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
      links: %{"GitHub" => "https://github.com/MLIR-China/beaver"}
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"},
      # {:sibling_app_in_umbrella, in_umbrella: true}
      {:beaver_capi, in_umbrella: true},
      {:beaver_dialect, in_umbrella: true},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end
end
