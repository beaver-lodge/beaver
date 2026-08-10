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

  @doc "Compile textual LLVM IR to NVPTX assembly in-process."
  @spec compile_to_ptx(binary(), keyword()) :: {:ok, binary()} | {:error, binary()}
  def compile_to_ptx(llvm_ir, opts \\ []) when is_binary(llvm_ir) and is_list(opts) do
    cpu = Keyword.get(opts, :cpu, "sm_80")
    features = Keyword.get(opts, :features, "")

    unless is_binary(cpu) and is_binary(features) do
      raise ArgumentError, ":cpu and :features must be strings"
    end

    MLIR.CAPI.beaver_raw_compile_llvm_ir_to_ptx(llvm_ir, cpu, features)
  end

  @doc "Compile textual LLVM IR to NVPTX assembly, raising on failure."
  @spec compile_to_ptx!(binary(), keyword()) :: binary()
  def compile_to_ptx!(llvm_ir, opts \\ []) do
    case compile_to_ptx(llvm_ir, opts) do
      {:ok, ptx} -> ptx
      {:error, message} -> raise ArgumentError, "failed to compile LLVM IR to PTX: #{message}"
    end
  end
end
