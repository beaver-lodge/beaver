defmodule Beaver.MLIR.CAPI.IR do
  @moduledoc false
  @deprecated "Use Beaver.MLIR.CAPI"
  use Exotic.Library
  @path Beaver.MLIR.CAPI
  @intptr_t :long

  def load!(), do: Exotic.load!(__MODULE__, Beaver.MLIR.CAPI)

  defmodule Dialect do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Context do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Location do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Module do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Operation do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Type do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Value do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Block do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Region do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Attribute do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule Identifier do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [ptr: :ptr]
  end

  defmodule NamedAttribute do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [name: Identifier, attribute: Attribute]
  end

  defmodule StringRef do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    use Exotic.Type.Struct, fields: [data: :ptr, length: :size]
  end

  defmodule OperationState do
    @moduledoc false
    @deprecated "Use Beaver.MLIR.CAPI"
    @intptr_t :i64
    use Exotic.Type.Struct,
      fields: [
        name: StringRef,
        location: Location,
        nResults: @intptr_t,
        results: :ptr,
        nOperands: @intptr_t,
        operands: :ptr,
        nRegions: @intptr_t,
        regions: :ptr,
        nSuccessors: @intptr_t,
        successors: :ptr,
        nAttributes: @intptr_t,
        attributes: :ptr,
        enableResultTypeInference: :bool
      ]
  end

  def mlirContextCreate(), do: Context
  def mlirContextDestroy(Context), do: :void
  def mlirLocationFileLineColGet(Context, :ptr, :i32, :i32), do: Location
  def mlirModuleCreateEmpty(Location), do: Module
  def mlirModuleGetOperation(Module), do: Operation
  def mlirModuleGetBody(Module), do: Block
  def mlirOperationDump(Operation), do: :void
  def mlirAttributeDump(Attribute), do: :void
  def mlirValueDump(Value), do: :void
  def mlirTypeDump(Type), do: :void
  def mlirOperationStateGet(StringRef, Location), do: OperationState
  def mlirStringRefCreateFromCString(:ptr), do: StringRef
  def mlirTypeParseGet(Location, StringRef), do: Type
  def mlirLocationUnknownGet(Context), do: Location
  def mlirRegionCreate(), do: Region
  def mlirBlockCreate(@intptr_t, :ptr, :ptr), do: Region
  def mlirBlockAddArgument(Block, Type, Location), do: Value
  def mlirBlockGetArgument(Block, @intptr_t), do: Value
  def mlirBlockInsertOwnedOperation(Block, @intptr_t, Operation), do: :void
  def mlirBlockInsertOwnedOperationAfter(Block, Operation, Operation), do: :void
  def mlirRegionAppendOwnedBlock(Region, Block), do: :void
  def mlirAttributeParseGet(Context, StringRef), do: Attribute
  def mlirIdentifierGet(Context, StringRef), do: Identifier
  def mlirIdentifierStr(Identifier), do: StringRef
  def mlirNamedAttributeGet(Identifier, Attribute), do: NamedAttribute
  def mlirOperationStateAddAttributes(:ptr, @intptr_t, :ptr), do: :void
  def mlirOperationStateAddOwnedRegions(:ptr, @intptr_t, :ptr), do: :void
  def mlirOperationStateAddOperands(:ptr, @intptr_t, :ptr), do: :void
  def mlirOperationStateAddResults(:ptr, @intptr_t, :ptr), do: :void
  def mlirOperationGetResult(Operation, @intptr_t), do: Value
  def mlirOperationGetNumResults(Operation), do: @intptr_t
  def mlirOperationVerify(Operation), do: :bool
  def mlirModuleCreateParse(Context, StringRef), do: Module
  def mlirDialectGetContext(Dialect), do: Context
  def mlirLocationGetContext(Location), do: Context
  def mlirModuleGetContext(Module), do: Context
  def mlirOperationGetContext(Operation), do: Context
  def mlirTypeGetContext(Type), do: Context
  def mlirAttributeGetContext(Attribute), do: Context
  def mlirIdentifierGetContext(Identifier), do: Context

  @doc """
  get a attr's type, note that it is not a `TypeAttr`
  """
  def mlirAttributeGetType(Attribute), do: Type
  def mlirAttributeEqual(Attribute, Attribute), do: :bool

  @doc """
  get the type a TypeAttr wraps
  """
  def mlirTypeAttrGetValue(Attribute), do: Type

  def mlirTypeIsAFunction(Type), do: :bool
  def mlirFunctionTypeGetNumInputs(Type), do: @intptr_t
  def mlirFunctionTypeGetNumResults(Type), do: @intptr_t
  def mlirFunctionTypeGetInput(Type, @intptr_t), do: Type
  def mlirFunctionTypeGetResult(Type, @intptr_t), do: Type

  def string_ref(value) when is_binary(value) do
    Exotic.Value.String.get(value)
    |> Exotic.Value.get_ptr()
    |> mlirStringRefCreateFromCString
  end

  def mlirOperationCreate(:ptr), do: Operation

  @native [
    mlirContextCreate: 0,
    mlirContextDestroy: 1,
    mlirLocationFileLineColGet: 4,
    mlirModuleCreateEmpty: 1,
    mlirModuleGetOperation: 1,
    mlirModuleGetBody: 1,
    mlirOperationDump: 1,
    mlirAttributeDump: 1,
    mlirValueDump: 1,
    mlirTypeDump: 1,
    mlirOperationStateGet: 2,
    mlirStringRefCreateFromCString: 1,
    mlirOperationCreate: 1,
    mlirTypeParseGet: 2,
    mlirRegionCreate: 0,
    mlirBlockCreate: 3,
    mlirBlockInsertOwnedOperation: 3,
    mlirBlockInsertOwnedOperationAfter: 3,
    mlirAttributeParseGet: 2,
    mlirRegionAppendOwnedBlock: 2,
    mlirIdentifierGet: 2,
    mlirIdentifierStr: 1,
    mlirNamedAttributeGet: 2,
    mlirOperationStateAddAttributes: 3,
    mlirOperationStateAddOwnedRegions: 3,
    mlirLocationUnknownGet: 1,
    mlirBlockAddArgument: 3,
    mlirBlockGetArgument: 2,
    mlirOperationStateAddOperands: 3,
    mlirOperationStateAddResults: 3,
    mlirOperationGetResult: 2,
    mlirOperationGetNumResults: 1,
    mlirModuleCreateParse: 2,
    mlirOperationVerify: 1,
    mlirDialectGetContext: 1,
    mlirLocationGetContext: 1,
    mlirModuleGetContext: 1,
    mlirOperationGetContext: 1,
    mlirTypeGetContext: 1,
    mlirAttributeGetContext: 1,
    mlirIdentifierGetContext: 1,
    mlirAttributeGetType: 1,
    mlirAttributeEqual: 2,
    mlirTypeAttrGetValue: 1,
    mlirTypeIsAFunction: 1,
    mlirFunctionTypeGetNumInputs: 1,
    mlirFunctionTypeGetNumResults: 1,
    mlirFunctionTypeGetInput: 2,
    mlirFunctionTypeGetResult: 2
  ]
end
