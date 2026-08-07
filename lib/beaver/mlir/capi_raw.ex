defmodule Beaver.MLIR.CAPI.Raw do
  @moduledoc false

  use Kinda.CodeGen,
    with: Beaver.MLIR.CAPI.CodeGen,
    root: Beaver.MLIR.CAPI,
    codec: Beaver.Native,
    surface: :raw

  @external_resource Beaver.MLIR.CAPI.CodeGen.declaration_manifest_path()

  for {name, arity} <- Beaver.MLIR.CAPI.Handwritten.functions() do
    args = Macro.generate_arguments(arity, __MODULE__)

    @doc false
    def unquote(name)(unquote_splicing(args)), do: :erlang.nif_error(:not_loaded)
  end

  @on_load :load_nif

  def load_nif do
    nif_file = nif_path()

    case :erlang.load_nif(nif_file, 0) do
      :ok -> :ok
      {:error, {:reload, _}} -> :ok
      {:error, reason} -> IO.puts("Failed to load nif: #{inspect(reason)}")
    end
  end

  defp nif_path do
    base = Path.join(:code.priv_dir(:beaver), "lib/libBeaverNIF")

    case :os.type() do
      {:win32, _} ->
        # Zig names Windows DLLs without the Unix "lib" prefix. OTP appends
        # the ".dll" suffix itself, so the extension must not be included in
        # the path passed to load_nif/2 (otherwise the loader looks for
        # BeaverNIF.dll.dll). The loader also resolves the DLL's sibling
        # dependencies only when the path uses backslashes: with forward
        # slashes, LOAD_WITH_ALTERED_SEARCH_PATH is not applied and the
        # standard search order misses priv/lib, failing with error 126.
        dll = Path.join(Path.dirname(base), "BeaverNIF")
        # Windows resolves a DLL's own dependencies against the search path,
        # not the directory of the DLL, so make the runtime libraries next to
        # the NIF discoverable before loading it.
        add_dll_search_path(Path.dirname(dll))
        dll |> String.replace("/", "\\") |> String.to_charlist()

      _ ->
        nif_file = String.to_charlist(base)
        dylib = "#{base}.dylib"

        if File.exists?(dylib) do
          dylib
          |> Path.basename()
          |> File.ln_s("#{base}.so")
        end

        nif_file
    end
  end

  defp add_dll_search_path(dir) do
    path = System.get_env("PATH") || ""
    sep = if match?({:win32, _}, :os.type()), do: ";", else: ":"

    unless path |> String.split(sep) |> Enum.any?(&(&1 == dir)) do
      System.put_env("PATH", dir <> sep <> path)
    end
  end
end
