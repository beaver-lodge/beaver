defmodule Beaver.MLIR.CAPI do
  @moduledoc """
  This module ships MLIR's C API. These NIFs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.

  ## MLIR CAPIs might trigger Elixir code execution
  Some MLIR CAPIs might trigger the execution of Elixir code by sending messages.
  Their respective NIFs will be created with dirty flag to prevent dead-locking the BEAM VM if the Elixir callback is scheduled to run on the same scheduler. That's why the Elixir callback shouldn't contain any code run on dirty scheduler. Also be aware of the performance of the Elixir callback, because when it is running, the dirty schedulers will be blocked to wait for a mutex.

  Here are the list of these MLIR CAPIs and the Elixir code to execute they might trigger:
  - `mlirPassManagerRunOnOp`: the MLIR pass implemented in Elixir.
  - `mlirOperationVerify`, `mlirAttributeParseGet`, `mlirTypeParseGet`, `mlirModuleCreateParse`: the diagnostic handler implemented in Elixir.
  """
  use Kinda.CodeGen, with: Beaver.MLIR.CAPI.CodeGen, root: __MODULE__, forward: Beaver.Native

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
        _destruct,
        _initialize,
        _clone,
        _run
      ),
      do: :erlang.nif_error(:not_loaded)

  def beaver_raw_run_pm_on_op_async(_pm, _op), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_destroy_pm_async(_pm), do: :erlang.nif_error(:not_loaded)

  def beaver_raw_logical_mutex_token_signal_success(_token, _is_success),
    do: :erlang.nif_error(:not_loaded)

  def beaver_raw_registered_ops(_ctx), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_registered_dialects(_ctx), do: :erlang.nif_error(:not_loaded)

  for f <- ~w{
    StringRef
    Attribute
    Type
    Operation
    OperationSpecialized
    OperationGeneric
    OperationBytecode
    Value
    AffineMap
    Location
    OpPassManager
    Identifier
    Diagnostic
  } do
    f = :"beaver_raw_to_string_#{f}"
    def unquote(f)(_), do: :erlang.nif_error(:not_loaded)
    {f, 1}
  end

  def beaver_raw_get_string_ref(_), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_read_opaque_ptr(_, _), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_deallocate_opaque_ptr(_), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_get_null_ptr(), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_context_attach_diagnostic_handler(_, _), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_jit_invoke_with_terms(_jit, _name, _args), do: :erlang.nif_error(:not_loaded)

  def beaver_raw_jit_invoke_with_terms_cpu_bound(_jit, _name, _args),
    do: :erlang.nif_error(:not_loaded)

  def beaver_raw_jit_invoke_with_terms_io_bound(_jit, _name, _args),
    do: :erlang.nif_error(:not_loaded)

  def beaver_raw_jit_register_enif(_jit), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_enif_signatures(_ctx), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_enif_functions(), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_mlir_type_of_enif_obj(_ctx, _obj), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_string_printer_callback(), do: :erlang.nif_error(:not_loaded)
  def beaver_raw_string_printer_flush(_sp), do: :erlang.nif_error(:not_loaded)
end
