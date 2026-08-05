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
    nif_file = ~c"#{:code.priv_dir(:beaver)}/lib/libBeaverNIF"
    dylib = "#{nif_file}.dylib"

    if File.exists?(dylib) do
      dylib
      |> Path.basename()
      |> File.ln_s("#{nif_file}.so")
    end

    case :erlang.load_nif(nif_file, 0) do
      :ok -> :ok
      {:error, {:reload, _}} -> :ok
      {:error, reason} -> IO.puts("Failed to load nif: #{inspect(reason)}")
    end
  end
end
