defmodule Beaver.MLIR.CAPI.Pass do
  @deprecated "Use Beaver.MLIR.CAPI"
  @moduledoc false
  use Exotic.Library
  @path "libMLIRPass.dylib"

  defmodule Pass do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule ExternalPass do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule PassManager do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule OpPassManager do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule ExternalPassCallbacks do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct,
      fields: [
        construct: :ptr,
        destruct: :ptr,
        initialize: :ptr,
        clone: :ptr,
        run: :ptr
      ]
  end

  alias Beaver.MLIR.CAPI.IR.{Context, Module, StringRef}
  alias Beaver.MLIR.CAPI.Support.{LogicalResult, TypeID}

  def mlirPassManagerCreate(Context), do: PassManager
  def mlirPassManagerDestroy(PassManager), do: :void
  def mlirPassManagerIsNull(PassManager), do: :bool
  def mlirPassManagerGetAsOpPassManager(PassManager), do: OpPassManager
  def mlirPassManagerRun(PassManager, Module), do: LogicalResult
  def mlirPassManagerEnableIRPrinting(PassManager), do: :void
  def mlirPassManagerEnableVerifier(PassManager, :bool), do: :void
  def mlirPassManagerGetNestedUnder(PassManager, StringRef), do: OpPassManager
  def mlirOpPassManagerGetNestedUnder(PassManager, StringRef), do: OpPassManager
  def mlirPassManagerAddOwnedPass(PassManager, Pass), do: :void
  def mlirOpPassManagerAddOwnedPass(OpPassManager, Pass), do: :void
  @string_callback :ptr
  @user_data :ptr
  def mlirPrintPassPipeline(OpPassManager, @string_callback, @user_data), do: :void
  def mlirParsePassPipeline(PassManager, StringRef), do: LogicalResult
  @intptr_t :i64
  def mlirCreateExternalPass(
        # passID
        TypeID,
        # name
        StringRef,
        # argument
        StringRef,
        # description
        StringRef,
        # opName
        StringRef,
        # nDependentDialects
        @intptr_t,
        # dependentDialects
        # DialectHandle.Pointer,
        :ptr,
        # callbacks
        ExternalPassCallbacks,
        # userData
        @user_data
      ),
      do: Pass

  def mlirExternalPassSignalFailure(ExternalPass), do: :void

  @native [
    mlirPassManagerCreate: 1,
    mlirPassManagerDestroy: 1,
    mlirPassManagerIsNull: 1,
    mlirPassManagerGetAsOpPassManager: 1,
    mlirPassManagerRun: 2,
    mlirPassManagerEnableIRPrinting: 1,
    mlirPassManagerEnableVerifier: 2,
    mlirPassManagerGetNestedUnder: 2,
    mlirOpPassManagerGetNestedUnder: 2,
    mlirPassManagerAddOwnedPass: 2,
    mlirOpPassManagerAddOwnedPass: 2,
    mlirPrintPassPipeline: 3,
    mlirParsePassPipeline: 2,
    mlirCreateExternalPass: 9,
    mlirExternalPassSignalFailure: 1
  ]
  def load!(), do: Exotic.load!(__MODULE__, Beaver.MLIR.CAPI)
end
