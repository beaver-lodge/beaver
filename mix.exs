defmodule Beaver.MixProject do
  use Mix.Project

  def project do
    [
      app: :beaver,
      version: "0.3.7-dev",
      elixir: "~> 1.9",
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
        make_force_build: System.get_env("BEAVER_BUILD_CMAKE") in ["1", "true"],
        make_precompiler_url:
          System.get_env("BEAVER_ARTEFACT_URL") ||
            "https://github.com/beaver-lodge/beaver-prebuilt/releases/download/2024-06-22-0327/@{artefact_filename}",
        make_precompiler_nif_versions: [
          versions: fn opts ->
            target = opts.target

            if String.contains?(target, "darwin") do
              ["2.17"]
            else
              ["2.16", "2.17"]
            end
          end
        ]
      ]
  end

  defp description() do
    "Beaver, a MLIR Toolkit in Elixir"
  end

  defp docs() do
    [
      main: "Beaver",
      extras: [
        "guides/your-first-beaver-compiler.livemd"
      ],
      filter_modules: fn m, _meta ->
        name = Atom.to_string(m)

        not (String.contains?(name, "Beaver.MLIR.Dialect.") ||
               String.contains?(name, "Beaver.MLIR.CAPI."))
      end,
      api_reference: false,
      groups_for_modules: [
        DSL: [
          Beaver,
          Beaver.Env,
          Beaver.Pattern,
          Beaver.Slang
        ],
        Walker: [
          ~r"Beaver.Walker.*"
        ],
        ENIF: [
          ~r"Beaver.ENIF.*"
        ],
        MLIR: [
          ~r"Beaver.MLIR.*"
        ],
        Native: [
          ~r"Beaver.Native.*"
        ]
      ]
    ]
  end

  defp package() do
    [
      licenses: ["Apache-2.0", "MIT"],
      links: %{"GitHub" => "https://github.com/beaver-lodge/beaver"},
      files: ~w{
        lib .formatter.exs mix.exs README*
        scripts/*.exs
        native/mlir-zig-proj/src/*.zig
        native/mlir-zig-proj/build.zig
        native/mlir-zig-proj/build.zig.zon
        native/**/CMakeLists.txt
        native/**/*.cmake
        native/**/*.h
        native/**/*.td
        native/**/*.cpp
        checksum.exs
        Makefile
      }
    ]
  end

  def application do
    [
      mod: {Beaver.Application, []},
      start_phases: [mlir_register_all_passes: []],
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:elixir_make, "~> 0.4", runtime: false},
      {:llvm_config, "~> 0.1.0"},
      if(p = System.get_env("BEAVER_KINDA_PATH"),
        do: {:kinda, path: p},
        else: {:kinda, "~> 0.8.1"}
      ),
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end
end
