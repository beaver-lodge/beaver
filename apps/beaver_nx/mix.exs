defmodule Beaver.Nx.MixProject do
  use Mix.Project

  def project do
    [
      app: :beaver_nx,
      version: "0.1.0",
      build_path: "../../_build",
      config_path: "../../config/config.exs",
      deps_path: "../../deps",
      lockfile: "../../mix.lock",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {Beaver.Nx.Application, []}
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.2"},
      {:beaver, in_umbrella: true},
      {:beaver_capi, in_umbrella: true}
    ]
  end
end
