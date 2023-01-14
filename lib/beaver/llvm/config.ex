defmodule Beaver.LLVM.Config do
  defp llvm_config_from_sys_env() do
    System.get_env("LLVM_CONFIG_PATH")
  end

  defp run_llvm_config(sub_cmd) do
    llvm_config = llvm_config_from_sys_env()

    if llvm_config do
      case System.cmd(llvm_config, [sub_cmd]) do
        {path, 0} ->
          {:ok, String.trim(path)}

        _ ->
          {:error, "failed to run llvm-config"}
      end
    else
      {:error, "LLVM_CONFIG_PATH is not set"}
    end
  end

  def include_dir() do
    run_llvm_config("--includedir")
  end

  def lib_dir() do
    run_llvm_config("--libdir")
  end
end
