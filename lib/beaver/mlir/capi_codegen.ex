defmodule Beaver.MLIR.CAPI.CodeGen do
  @moduledoc false
  alias Kinda.CodeGen.{KindDecl}
  use Kinda.CodeGen

  defp memref_kind_functions(DescriptorUnranked) do
    [
      make: 5,
      aligned: 1,
      allocated: 1,
      offset: 1
    ]
  end

  defp memref_kind_functions(_) do
    [
      make: 5,
      aligned: 1,
      allocated: 1,
      offset: 1,
      sizes: 1,
      strides: 1
    ]
  end

  @impl true
  def kinds() do
    for rank <- [
          DescriptorUnranked,
          Descriptor1D,
          Descriptor2D,
          Descriptor3D,
          Descriptor4D,
          Descriptor5D,
          Descriptor6D,
          Descriptor7D,
          Descriptor8D,
          Descriptor9D
        ],
        t <- [Complex.F32, U8, U16, U32, I8, I16, I32, I64, F32, F64] do
      %KindDecl{
        module_name: Module.concat([Beaver.Native, t, MemRef, rank]),
        kind_functions: memref_kind_functions(rank)
      }
    end ++
      [
        %KindDecl{
          module_name: Beaver.Native.PtrOwner
        },
        %KindDecl{
          module_name: Beaver.Native.Complex.F32
        }
      ] ++
      Enum.map(
        [
          :Type,
          :Pass,
          :LogicalResult,
          :StringRef,
          :Context,
          :Location,
          :Attribute,
          :Operation,
          :AffineMap,
          :DiagnosticHandlerDeleteUserData,
          :NamedAttribute,
          :Region,
          :Module,
          :GenericCallback,
          :ExternalPassConstruct,
          :ExternalPassRun,
          :Identifier,
          :Value,
          :Block,
          :Dialect,
          :SymbolTableWalkSymbolTablesCallback,
          :OpOperand,
          :AffineMapCompressUnusedSymbolsPopulateResult,
          :UnmanagedDenseResourceElementsAttrGetDeleteCallback
        ],
        &%KindDecl{module_name: Module.concat(Beaver.MLIR, &1)}
      ) ++
      Enum.map(
        [
          :Enum_MlirSparseTensorLevelType,
          :MlirShapedTypeComponentsCallback,
          :MlirTypeID,
          :MlirTypesCallback,
          :MlirIntegerSet,
          :MlirAffineExpr,
          :MlirStringCallback,
          :MlirDialectHandle,
          :MlirDialectRegistry,
          :MlirDiagnosticHandlerID,
          :MlirDiagnosticHandler,
          :MlirDiagnostic,
          :MlirDiagnosticSeverity,
          :MlirPassManager,
          :MlirRewritePatternSet,
          :MlirExecutionEngine,
          :MlirOperationState,
          :MlirSymbolTable,
          :MlirRegisteredOperationName,
          :MlirExternalPass,
          :MlirExternalPassCallbacks,
          :MlirOpPassManager,
          :MlirAsmState,
          :MlirOperationWalkCallback,
          :MlirWalkOrder,
          :MlirBytecodeWriterConfig,
          :MlirOpPrintingFlags,
          :MlirLlvmThreadPool,
          :MlirTypeIDAllocator
        ],
        &%KindDecl{module_name: Module.concat(Beaver.MLIR.CAPI, &1)}
      ) ++
      Enum.map(
        [
          :ISize,
          :OpaquePtr,
          :Bool,
          :CInt,
          :F64,
          :I32,
          :I64,
          :CUInt,
          :F32,
          :U64,
          :U32,
          :U16,
          :I8,
          :I16,
          :U8,
          :USize,
          :OpaqueArray
        ],
        &%KindDecl{module_name: Module.concat(Beaver.Native, &1)}
      )
  end
end
