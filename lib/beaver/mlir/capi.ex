defmodule Beaver.MLIR.CAPI do
  @moduledoc """
  This module ships MLIR's C API. These NIFs are generated from headers in LLVM repo and this repo's headers providing supplemental functions.

  ## MLIR CAPIs might trigger Elixir code execution
  Some MLIR CAPIs might trigger Elixir callbacks through a context worker.
  Their entry points use an asynchronous or dirty-scheduler boundary so a
  normal BEAM scheduler never waits for a callback that it must itself run.
  Callback implementations must not synchronously re-enter the same callback
  attachment.

  Here are the list of these MLIR CAPIs and the Elixir code to execute they might trigger:
  - `mlirPassManagerRunOnOp`: the MLIR pass implemented in Elixir.
  - `mlirTransformApplyNamedSequence`: Transform operations implemented in Elixir.
  - `mlirOperationVerify`, `mlirAttributeParseGet`, `mlirTypeParseGet`, `mlirModuleCreateParse`: native diagnostic collection and any external operation interface reached during parsing or verification.
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
