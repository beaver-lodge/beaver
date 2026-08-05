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
  use Kinda.CodeGen,
    with: Beaver.MLIR.CAPI.CodeGen,
    root: __MODULE__,
    raw_module: __MODULE__.Raw,
    codec: Beaver.Native,
    surface: :public

  # Recompile the Elixir surface only when the generated C API changes. Native
  # implementation files are tracked independently by the native build.
  @external_resource Beaver.MLIR.CAPI.CodeGen.declaration_manifest_path()

  for {name, arity} <- Beaver.MLIR.CAPI.Handwritten.functions() do
    args = Macro.generate_arguments(arity, __MODULE__)
    call = {{:., [], [__MODULE__.Raw, name]}, [], args}

    def unquote(name)(unquote_splicing(args)), do: unquote(call)
  end

  defdelegate load_nif(), to: __MODULE__.Raw
end
