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
      compilers: [:cmake] ++ Mix.compilers(),
      aliases: aliases()
    ]
  end

  defp description() do
    "Beaver, a MLIR Toolkit in Elixir"
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
      {:fizz, in_umbrella: true},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
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

  defp cmake(_) do
    cmake_project = "native/mlir-c"
    build = Path.join(Mix.Project.build_path(), "mlir-c-build")
    install = Path.join(Mix.Project.build_path(), "mlir-c-install")

    IO.puts("[CMake] configuring...")

    {_, 0} =
      System.cmd("cmake", [
        "-S",
        cmake_project,
        "-B",
        build,
        "-G",
        "Ninja",
        "-DLLVM_DIR=#{LLVM.Config.lib_dir()}/cmake/llvm",
        "-DMLIR_DIR=#{LLVM.Config.lib_dir()}/cmake/mlir",
        "-DCMAKE_INSTALL_PREFIX=#{install}"
      ])

    IO.puts("[CMake] building...")

    with {_, 0} <- System.cmd("cmake", ["--build", build, "--target", "install"]) do
      IO.puts("[CMake] done")
      :ok
    else
      {error, _} ->
        IO.puts(error)
        {:error, [error]}
    end
  end
end
