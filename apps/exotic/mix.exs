defmodule Exotic.MixProject do
  use Mix.Project

  def project do
    [
      app: :exotic,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      mod: {Exotic.Application, []},
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:rustler, "~> 0.25.0"}
    ]
  end
end
