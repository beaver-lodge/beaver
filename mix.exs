defmodule Beaver.MixProject do
  use Mix.Project

  @build_cmake Application.compile_env(
                 :beaver,
                 :build_cmake,
                 System.get_env("BEAVER_BUILD_CMAKE") in ["1", "true"]
               )

  def project do
    make_compilers =
      if @build_cmake do
        [:elixir_make]
      else
        []
      end

    [
      app: :beaver,
      version: "0.3.2",
      elixir: "~> 1.9",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description: description(),
      docs: docs(),
      package: package(),
      compilers: make_compilers ++ Mix.compilers(),
      preferred_cli_env: [
        "test.watch": :test
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
        native/mlir-zig-proj/src/*.zig
        native/mlir-zig-proj/build.zig
        native/mlir-zig-proj/build.zig.zon
        native/**/CMakeLists.txt
        native/**/*.cmake
        native/**/*.h
        native/**/*.td
        native/**/*.cpp
        checksum-*.exs
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
        else: {:kinda, "~> 0.3.0"}
      ),
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      {:mix_test_watch, "~> 1.0", only: [:dev, :test]},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false},
      {:gradient, github: "esl/gradient", only: [:dev], runtime: false},
      {:doctor, "~> 0.21.0", only: :dev}
    ]
  end
end
