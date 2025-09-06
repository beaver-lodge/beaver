defmodule Beaver.MixProject do
  use Mix.Project

  def project do
    [
      app: :beaver,
      version: "0.4.7-dev",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description: description(),
      docs: docs(),
      package: package(),
      compilers: [:elixir_make] ++ Mix.compilers()
    ] ++
      [
        make_precompiler: {:nif, Kinda.Precompiler},
        make_force_build: System.get_env("BEAVER_MAKE_FORCE_BUILD") in ["1", "true"],
        make_precompiler_url:
          System.get_env("BEAVER_ARTEFACT_URL") ||
            "https://github.com/beaver-lodge/beaver-prebuilt/releases/download/2025-09-05-1313/@{artefact_filename}",
        make_precompiler_nif_versions: [
          versions: fn _ -> ["2.16", "2.17"] end
        ],
        make_args: ~w{-j},
        make_cwd: "native",
        make_clean: ["clean"]
      ]
  end

  defp description() do
    "Beaver, a MLIR Toolkit in Elixir"
  end

  defp docs() do
    [
      main: "Beaver",
      extras: [
        "guides/your-first-beaver-compiler.livemd",
        "CONTRIBUTING.md",
        "README.md"
      ],
      filter_modules: fn m, _meta ->
        name = Atom.to_string(m)
        not String.contains?(name, "Beaver.MLIR.CAPI.")
      end,
      api_reference: false,
      groups_for_modules: [
        DSL: [
          Beaver,
          Beaver.Env,
          Beaver.Pattern,
          Beaver.Slang,
          ~r"Beaver.Exterior.*",
          Beaver.SSA
        ],
        Utils: [
          ~r"Beaver.Walker.*",
          ~r"Beaver.Deferred.*",
          ~r"Beaver.Pass.*",
          ~r"Beaver.Printer.*",
          ~r"Beaver.Composer.*",
          ~r"Beaver.Sigils.*"
        ],
        ENIF: [
          ~r"Beaver.ENIF.*"
        ],
        MLIR: [
          ~r"Beaver.MLIR(?!\.Dialect).*"
        ],
        Dialect: [
          ~r"Beaver.MLIR.Dialect.*"
        ],
        Native: [
          ~r"Beaver.Native.*"
        ]
      ]
    ]
  end

  @external_files __DIR__ |> Path.join("external_files.txt") |> File.read!()
  defp package() do
    [
      licenses: ["Apache-2.0", "MIT"],
      links: %{"GitHub" => "https://github.com/beaver-lodge/beaver"},
      files: ~w{
        lib .formatter.exs mix.exs README*
        checksum.exs
        external_files.txt
        #{@external_files}
      }
    ]
  end

  def application do
    [
      mod: {Beaver.Application, []},
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:dev), do: ~w{lib bench profile}
  defp elixirc_paths(:test), do: ~w{lib bench profile test/support}
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:elixir_make, "~> 0.4", runtime: false},
      {:kinda, "~> 0.10.3"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      {:benchee, "~> 1.0", only: :dev},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end
end
