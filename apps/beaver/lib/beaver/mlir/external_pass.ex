defmodule Beaver.MLIR.ExternalPass do
  @doc """
  Lower level API to work with MLIR's external pass (pass defined in C). Use Beaver.MLIR.Pass for idiomatic Erlang behavior.
  """
  defstruct external: nil
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI
  require Beaver.MLIR.CAPI

  @doc """
  Create a pass by passing a callback module
  """
  def create(pass_module, typeIDAllocator, op_name \\ "") do
    description = MLIR.StringRef.create("")
    emptyOpName = MLIR.StringRef.create(op_name)
    passID = CAPI.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator)
    name = Atom.to_string(pass_module) |> CAPI.c_string() |> CAPI.mlirStringRefCreateFromCString()
    argument = CAPI.c_string("test-external-pass") |> CAPI.mlirStringRefCreateFromCString()

    callbacks =
      %CAPI.MlirExternalPassCallbacks{} =
      Exotic.Value.Struct.get(
        CAPI.MlirExternalPassCallbacks,
        [
          __MODULE__,
          __MODULE__,
          # we should only expose run and initialize is optional, so pass a null ptr
          Exotic.Value.Ptr.null(),
          __MODULE__,
          pass_module
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

  def handle_invoke(:construct, [user_data_ptr], state) do
    {:return, user_data_ptr, state}
  end

  def handle_invoke(:destruct, [user_data_ptr], state) do
    {:return, user_data_ptr, state}
  end

  def handle_invoke(:initialize, [%MLIR.CAPI.MlirContext{}, user_data_ptr], state) do
    {:return, user_data_ptr, state}
  end

  def handle_invoke(:clone, [_user_data_ptr], state) do
    {:pass, state}
  end
end
