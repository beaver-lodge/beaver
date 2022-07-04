defmodule Beaver.MLIR.ExternalPass do
  @doc """
  Lower level API to work with MLIR's external pass (pass defined in C). Use Beaver.MLIR.Pass for idiomatic Erlang behavior.
  """
  defstruct external: nil
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  defmodule Callbacks do
    @moduledoc """
    Callbacks are used to implement a pass. Each field of the struct is a closure handler process pid.
    """

    def get(
          construct: construct,
          destruct: destruct,
          initialize: initialize,
          clone: clone,
          run: run
        ) do
      %CAPI.MlirExternalPassCallbacks{} =
        Exotic.Value.Struct.get(
          CAPI.MlirExternalPassCallbacks,
          [
            construct,
            destruct,
            initialize,
            clone,
            run
          ]
        )
    end
  end

  @doc """
  Create a pass by passing a callback module
  """
  def create(callback_module, user_state, typeIDAllocator, op_name \\ "") do
    description = MLIR.StringRef.create("")
    emptyOpName = MLIR.StringRef.create(op_name)
    passID = CAPI.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator)
    name = CAPI.mlirStringRefCreateFromCString("TestExternalFuncPass")
    argument = CAPI.mlirStringRefCreateFromCString("test-external-pass")

    callbacks =
      Callbacks.get(
        construct: callback_module,
        destruct: callback_module,
        initialize: callback_module,
        clone: callback_module,
        run: callback_module
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
