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
