defmodule Beaver.MLIR.CompilationRuntime do
  @moduledoc "LLVM build identity used by external compiler-kernel manifests."
  alias Beaver.MLIR

  @spec llvm_revision() :: String.t()
  def llvm_revision, do: MLIR.CAPI.beaverGetLLVMVersion() |> MLIR.to_string()
end
