defmodule Beaver.MixProject do
  use Mix.Project

  def project do
    [
      app: :beaver,
      version: "0.2.17",
      elixir: "~> 1.9",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      description: description(),
      docs: docs(),
      package: package(),
      compilers: [:cmake] ++ Mix.compilers(),
      aliases: aliases(),
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
          Beaver.Pattern
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
        native/mlir-zig-src/*.zig
        native/mlir-zig-proj/prod/build.zig
        native/mlir-zig-proj/dev/build.zig
        native/mlir-zig-proj/test/build.zig
        native/**/CMakeLists.txt
        native/**/*.cmake
        native/**/*.h
        native/**/*.td
        native/**/*.cpp
        checksum-*.exs
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
      {:llvm_config, "~> 0.1.0"},
      {:kinda, "~> 0.2.0"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      {:mix_test_watch, "~> 1.0", only: [:dev, :test]},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false},
      {:gradient, github: "esl/gradient", only: [:dev], runtime: false},
      {:doctor, "~> 0.21.0", only: :dev}
    ]
  end

  defp aliases do
    [
      "compile.cmake": &compile_cmake/1,
      "beaver.cmake": &beaver_cmake/1
    ]
  end

  require Logger

  defp do_cmake() do
    cmake_project = "native/mlir-c"
    build = Path.join(Mix.Project.app_path(), "mlir-c-build")
    install = Path.join(Mix.Project.app_path(), "native-install")

    Logger.debug("[CMake] running...")

    {:ok, llvm_lib_dir} = LLVMConfig.lib_dir()

    llvm_cmake_dir = Path.join(llvm_lib_dir, "cmake/llvm")
    mlir_cmake_dir = Path.join(llvm_lib_dir, "cmake/mlir")

    with {_, 0} <-
           System.cmd(
             "cmake",
             [
               "-S",
               cmake_project,
               "-B",
               build,
               "-G",
               "Ninja",
               "-DLLVM_DIR=#{llvm_cmake_dir}",
               "-DMLIR_DIR=#{mlir_cmake_dir}",
               "-DCMAKE_INSTALL_PREFIX=#{install}"
             ],
             stderr_to_stdout: true
           ),
         {_, 0} <-
           System.cmd("cmake", ["--build", build, "--target", "install"], stderr_to_stdout: true) do
      Logger.debug("[CMake] installed to #{install}")
      :ok
    else
      {error, _} ->
        Logger.info(error)
        {:error, [error]}
    end
  end

  @build_cmake Application.compile_env(:beaver, :build_cmake, false)
  defp compile_cmake(args) do
    if @build_cmake or "--force" in args do
      do_cmake()
    else
      :noop
    end
  end

  defp beaver_cmake(args) do
    Mix.Tasks.Deps.Compile.run(args)
    :ok = Application.ensure_started(:llvm_config)
    Code.ensure_compiled!(LLVMConfig)
    do_cmake()
  end
end
