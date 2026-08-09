defmodule Beaver.MLIR.Target.LLVMIR do
  @moduledoc """
  Safe export of fully lowered MLIR modules to textual LLVM IR.

  The native adapter owns and releases the temporary LLVM context, module and
  printed message. LLVM opaque handles never cross the NIF boundary.
  """

  alias Beaver.MLIR

  @spec translate(MLIR.Module.t()) :: {:ok, binary()} | {:error, MLIR.diagnostics()}
  def translate(%MLIR.Module{} = module) do
    module
    |> MLIR.context()
    |> MLIR.Context.register_translations()

    module
    |> MLIR.Operation.from_module()
    |> then(&MLIR.CAPI.beaver_raw_translate_module_to_llvm_ir(&1.ref))
    |> Beaver.Native.check!()
    |> case do
      {llvm_ir, _diagnostics} when is_binary(llvm_ir) -> {:ok, llvm_ir}
      {:error, diagnostics} when is_list(diagnostics) -> {:error, diagnostics}
    end
  end

  @spec translate!(MLIR.Module.t()) :: binary()
  def translate!(%MLIR.Module{} = module) do
    case translate(module) do
      {:ok, llvm_ir} ->
        llvm_ir

      {:error, diagnostics} ->
        raise ArgumentError,
              MLIR.Diagnostic.format(diagnostics, "failed to translate module to LLVM IR")
    end
  end
end
