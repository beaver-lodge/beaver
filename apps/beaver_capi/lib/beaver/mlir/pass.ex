defmodule Beaver.MLIR.Pass do
  defstruct external: nil
  alias Beaver.MLIR
  alias Beaver.MLIR.CAPI

  defmacro __using__(_) do
    quote([]) do
      @behaviour MLIR.Pass
      def do_something(), do: "Didn't do much but still"
      defoverridable do_something: 0
    end
  end

  # TODO: make callbacks by adding `defoverridable` in an using
  @callback init(map()) :: map()
  @callback construct(map()) :: map()
  @callback run(map(), map()) :: map()
  @callback destruct(map()) :: map()
  @callback clone(map()) :: map()

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

    externalPass =
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

    %__MODULE__{external: externalPass}
  end

  def pipeline!(pm, pipeline_str) when is_binary(pipeline_str) do
    status = CAPI.mlirParsePassPipeline(pm, MLIR.StringRef.create(pipeline_str))

    if not MLIR.LogicalResult.success?(status) do
      raise "Unexpected failure parsing pipeline: #{pipeline_str}"
    end

    pm
  end
end
