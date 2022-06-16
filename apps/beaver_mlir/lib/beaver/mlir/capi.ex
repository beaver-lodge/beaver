defmodule Beaver.LLVM.Config do
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
end

defmodule Beaver.MLIR.CAPI do
  @moduledoc """
  This module calls C API of MLIR. These FFIs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.
  """
  use Exotic.Library

  def load!() do
    # call to load code
    Beaver.MLIR.NIF.add(1, 2)

    path = Beaver.MLIR.NIF.load_from_path()
    path = Path.join(Application.app_dir(:beaver_mlir), path)
    lib_path = Path.wildcard("#{path}*") |> List.first()
    Exotic.load!(__MODULE__, lib_path)
  end

  paths = Path.wildcard("native/mlir_nif/met/include/**/*.h")

  for path <- paths do
    @external_resource path
  end

  wrapper_header_path = "include/wrapper/llvm/14.h"
  @external_resource wrapper_header_path

  @include %Exotic.Header{
    file: wrapper_header_path,
    search_paths: [
      Beaver.LLVM.Config.include_dir(),
      Path.join(File.cwd!(), "native/mlir_nif/met/include")
    ]
  }
  # TODO: fix me
  def call_to_load_code do
  end
end
