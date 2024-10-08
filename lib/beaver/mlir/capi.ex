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

  # setting up elixir re-compilation triggered by changes in external files
  for path <-
        ~w{#{Mix.Project.project_file() |> Path.dirname() |> Path.join("external_files.txt") |> File.read!()}}
        |> Enum.flat_map(&Path.wildcard/1) do
    @external_resource path
  end

  # stubs for hand-written NIFs
  def beaver_raw_create_mlir_pass(
        _name,
        _argument,
        _description,
        _op_name,
        _handler
      ),
      do: raise("NIF not loaded")

  def beaver_raw_pass_token_signal(_), do: raise("NIF not loaded")
  def beaver_raw_registered_ops(_ctx), do: raise("NIF not loaded")
  def beaver_raw_registered_dialects(_ctx), do: raise("NIF not loaded")
  def beaver_raw_string_ref_to_binary(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_attribute(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_type(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_operation(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_operation_specialized(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_operation_generic(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_operation_bytecode(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_value(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_affine_map(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_location(_), do: raise("NIF not loaded")
  def beaver_raw_to_string_pm(_), do: raise("NIF not loaded")
  def beaver_raw_get_string_ref(_), do: raise("NIF not loaded")
  def beaver_raw_read_opaque_ptr(_, _), do: raise("NIF not loaded")
  def beaver_raw_deallocate_opaque_ptr(_), do: raise("NIF not loaded")
  def beaver_raw_get_null_ptr(), do: raise("NIF not loaded")
  def beaver_raw_context_attach_diagnostic_handler(_, _), do: raise("NIF not loaded")
  def beaver_raw_get_diagnostic_string_callback(), do: raise("NIF not loaded")
  def beaver_raw_jit_invoke_with_terms(_jit, _name, _args), do: raise("NIF not loaded")
  def beaver_raw_jit_invoke_with_terms_cpu_bound(_jit, _name, _args), do: raise("NIF not loaded")
  def beaver_raw_jit_invoke_with_terms_io_bound(_jit, _name, _args), do: raise("NIF not loaded")
  def beaver_raw_jit_register_enif(_jit), do: raise("NIF not loaded")
  def beaver_raw_enif_signatures(_ctx), do: raise("NIF not loaded")
  def beaver_raw_enif_functions(), do: raise("NIF not loaded")
  def beaver_raw_mlir_type_of_enif_obj(_ctx, _obj), do: raise("NIF not loaded")
end
