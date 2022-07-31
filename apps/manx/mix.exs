defmodule Manx.MixProject do
  use Mix.Project

  def project do
    [
      app: :manx,
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
      mod: {Manx.Application, []}
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.2"},
      {:beaver, in_umbrella: true}
    ]
  end
end
