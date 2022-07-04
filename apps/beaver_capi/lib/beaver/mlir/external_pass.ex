defmodule Beaver.MLIR.ExternalPass do
  @doc """
  Lower level API to work with MLIR's external pass (pass defined in C). Use Beaver.MLIR.Pass for idiomatic Erlang behavior.
  """
  defstruct external: nil
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  @doc """
  Create a pass by passing a callback module
  """
  def create(callback_module, typeIDAllocator, op_name \\ "") do
    description = MLIR.StringRef.create("")
    emptyOpName = MLIR.StringRef.create(op_name)
    passID = CAPI.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator)
    name = CAPI.mlirStringRefCreateFromCString("TestExternalFuncPass")
    argument = CAPI.mlirStringRefCreateFromCString("test-external-pass")

    callbacks =
      %CAPI.MlirExternalPassCallbacks{} =
      Exotic.Value.Struct.get(
        CAPI.MlirExternalPassCallbacks,
        [
          callback_module,
          callback_module,
          callback_module,
          callback_module,
          callback_module
        ]
      )

    CAPI.mlirCreateExternalPass(
      passID,
      name,
      argument,
      description,
      emptyOpName,
      0,
      Exotic.Value.Ptr.null(),
      callbacks,
      Exotic.Value.Ptr.null()
    )
  end
end
