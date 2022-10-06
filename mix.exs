defmodule Beaver.MixProject do
  use Mix.Project

  def project do
    [
      app: :beaver,
      version: "0.2.5",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description: description(),
      docs: docs(),
      package: package(),
      compilers: [:cmake] ++ Mix.compilers(),
      aliases: aliases()
    ]
  end

  defp description() do
    "Beaver, a MLIR Toolkit in Elixir"
  end

  defp docs() do
    [
      main: "Beaver",
      extras: [
        "guides/introducing-beaver.md",
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
          ~r"Beaver.DSL.*",
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
      links: %{"GitHub" => "https://github.com/beaver-project/beaver"},
      files: ~w(lib priv .formatter.exs mix.exs README* native checksum-*.exs)
    ]
  end

  def application do
    [
      mod: {Beaver.Application, []},
      start_phases: [load_dialect_modules: []],
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:kinda, "~> 0.2"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      {:quark, "~> 2.3"}
    ]
  end

  defp aliases do
    ["compile.cmake": &cmake/1]
  end

  defmodule LLVM.Config do
    def include_dir() do
      llvm_config = System.get_env("LLVM_CONFIG_PATH", "llvm-config")

      path =
        with {path, 0} <- System.cmd(llvm_config, ["--includedir"]) do
          path
        else
          _ ->
            with {path, 0} <- System.shell("llvm-config-15 --includedir") do
              path
            else
              _ ->
                raise "fail to run llvm-config"
            end
        end

      path |> String.trim()
    end

    def lib_dir() do
      llvm_config = System.get_env("LLVM_CONFIG_PATH", "llvm-config")

      path =
        with {path, 0} <- System.cmd(llvm_config, ["--libdir"]) do
          path
        else
          _ ->
            with {path, 0} <- System.shell("llvm-config-15 --libdir") do
              path
            else
              _ ->
                raise "fail to run llvm-config"
            end
        end

      path |> String.trim()
    end
  end

  require Logger

  defp do_cmake() do
    cmake_project = "native/mlir-c"
    build = Path.join(Mix.Project.build_path(), "mlir-c-build")
    install = Path.join(Mix.Project.build_path(), "native-install")

    Logger.debug("[CMake] configuring...")

    {_, 0} =
      System.cmd(
        "cmake",
        [
          "-S",
          cmake_project,
          "-B",
          build,
          "-G",
          "Ninja",
          "-DLLVM_DIR=#{LLVM.Config.lib_dir()}/cmake/llvm",
          "-DMLIR_DIR=#{LLVM.Config.lib_dir()}/cmake/mlir",
          "-DCMAKE_INSTALL_PREFIX=#{install}"
        ],
        stderr_to_stdout: true
      )

    Logger.debug("[CMake] building...")

    with {_, 0} <- System.cmd("cmake", ["--build", build, "--target", "install"]) do
      Logger.debug("[CMake] installed to #{install}")
      :ok
    else
      {error, _} ->
        Logger.error(error)
        {:error, [error]}
    end
  end

  defp cmake(args) do
    if "--force" in args do
      do_cmake()
    else
      :noop
    end
  end
end
