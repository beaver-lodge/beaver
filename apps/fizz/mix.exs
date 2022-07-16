defmodule Fizz.MixProject do
  use Mix.Project

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

  def project do
    [
      app: :fizz,
      version: "0.1.0",
      build_path: "../../_build",
      config_path: "../../config/config.exs",
      deps_path: "../../deps",
      lockfile: "../../mix.lock",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: [:cmake] ++ Mix.compilers(),
      aliases: aliases()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"},
      # {:sibling_app_in_umbrella, in_umbrella: true}
    ]
  end

  defp aliases do
    ["compile.cmake": &cmake/1]
  end

  defp cmake(_) do
    cmake_project = "../../apps/beaver/native/mlir_nif/met"
    build = Path.join(Mix.Project.build_path(), "cmake_build")
    # install = Path.join(Mix.Project.build_path(), "cmake_install")
    install = "capi/zig-out/"

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

    {_, 0} = System.cmd("cmake", ["--build", build, "--target", "install"])
    :ok
  end
end
