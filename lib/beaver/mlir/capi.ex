defmodule Beaver.MLIR.CAPI do
  @moduledoc """
  This module ships MLIR's C API. These NIFs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.
  """
  use Kinda.CodeGen, with: Beaver.MLIR.CAPI.CodeGen, root: __MODULE__, forward: Beaver.Native

  @on_load :load_nif

  def load_nif do
    nif_file = ~c"#{:code.priv_dir(:beaver)}/lib/libBeaverNIF"

    if File.exists?(dylib = "#{nif_file}.dylib") do
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

  llvm_headers =
    case LLVMConfig.include_dir() do
      {:ok, include_dir} ->
        include_dir
        |> Path.join("*.h")
        |> Path.wildcard()

      _ ->
        []
    end

  # setting up elixir re-compilation triggered by changes in external files
  for path <-
        llvm_headers ++
          Path.wildcard("native/mlir-c/**/*.h") ++
          Path.wildcard("native/mlir-c/**/*.cpp") ++
          Path.wildcard("native/mlir-zig-proj/**/*.zig") ++
          Path.wildcard("native/mlir-zig-proj/**/*.zon"),
      not String.contains?(path, "kinda.gen.zig") do
    @external_resource path
  end

  # stubs for hand-written NIFs
  def beaver_raw_get_context_load_all_dialects(), do: raise("NIF not loaded")

  def beaver_raw_create_mlir_pass(
        _name,
        _argument,
        _description,
        _op_name,
        _handler
      ),
      do: raise("NIF not loaded")

  def beaver_raw_pass_token_signal(_), do: raise("NIF not loaded")
  def beaver_raw_registered_ops(), do: raise("NIF not loaded")
  def beaver_raw_registered_ops_of_dialect(_ctx, _name), do: raise("NIF not loaded")
  def beaver_raw_registered_dialects(), do: raise("NIF not loaded")
  def beaver_raw_resource_c_string_to_term_charlist(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_attribute(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_type(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_operation(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_operation_specialized(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_value(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_affine_map(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_location(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_pm(_), do: raise("NIF not loaded")
  def beaver_raw_mlir_named_attribute_get(_, _), do: raise("NIF not loaded")
  def beaver_raw_get_resource_c_string(_), do: raise("NIF not loaded")
  def beaver_raw_read_opaque_ptr(_, _), do: raise("NIF not loaded")
  def beaver_raw_own_opaque_ptr(_), do: raise("NIF not loaded")
  def beaver_raw_context_attach_diagnostic_handler(_, _), do: raise("NIF not loaded")
  def beaver_raw_parse_pass_pipeline(_, _), do: raise("NIF not loaded")
  def mif_raw_jit_invoke_with_terms(_jit, _name, _args), do: raise("NIF not loaded")
  def mif_raw_jit_register_enif(_jit), do: raise("NIF not loaded")
  def mif_raw_enif_signatures(_ctx), do: raise("NIF not loaded")
  def mif_raw_enif_functions(), do: raise("NIF not loaded")
  def mif_raw_mlir_type_of_enif_obj(_ctx, _obj), do: raise("NIF not loaded")
end
