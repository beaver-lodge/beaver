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
      package: package(),
      compilers: [:beaver_setup] ++ Mix.compilers() ++ [:beaver_teardown],
      aliases: aliases()
    ]
  end

  defp description() do
    "Beaver, a MLIR Toolkit in Elixir"
  end

  defp aliases do
    ["compile.beaver_setup": &setup/1, "compile.beaver_teardown": &teardown/1]
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
      start_phases: [load_dialect_modules: []],
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:exotic, in_umbrella: true},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  defp setup(_) do
    Beaver.MLIR.CAPI.Managed.start_link([])

    :ok
  end

  defp teardown(_) do
    pid = Process.whereis(Beaver.MLIR.CAPI.Managed)
    Process.unlink(pid)
    Process.exit(pid, :kill)

    :ok
  end
end
