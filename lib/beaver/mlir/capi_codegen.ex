defmodule Beaver.MLIR.CAPI.CodeGen do
  @moduledoc false
  alias Kinda.CodeGen.{KindDecl}
  @behaviour Kinda.CodeGen
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

  @impl Kinda.CodeGen
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
          :UnmanagedDenseResourceElementsAttrGetDeleteCallback,
          :SparseTensorLevelType,
          :ShapedTypeComponentsCallback,
          :TypeID,
          :TypesCallback,
          :IntegerSet,
          :AffineExpr,
          :StringCallback,
          :DialectHandle,
          :DialectRegistry,
          :DiagnosticHandlerID,
          :DiagnosticHandler,
          :Diagnostic,
          :DiagnosticSeverity,
          :PassManager,
          :RewritePatternSet,
          :ExecutionEngine,
          :"Operation.State",
          :SymbolTable,
          :ExternalPass,
          :ExternalPassCallbacks,
          :OpPassManager,
          :AsmState,
          :OperationWalkCallback,
          :WalkOrder,
          :BytecodeWriterConfig,
          :OpPrintingFlags,
          :LLVMThreadPool,
          :TypeIDAllocator,
          :RewriterBase,
          :FrozenRewritePatternSet,
          :PDLPatternModule,
          :GreedyRewriteDriverConfig
        ],
        &%KindDecl{module_name: Module.concat(Beaver.MLIR, &1)}
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
          :OpaqueArray,
          :StringArray
        ],
        &%KindDecl{module_name: Module.concat(Beaver.Native, &1)}
      ) ++ [%KindDecl{module_name: Beaver.Printer, kind_functions: [make: 0]}]
  end

  @impl Kinda.CodeGen
  def nifs() do
    Application.app_dir(:beaver)
    |> Path.join("priv/capi_functions.ex")
    |> File.read!()
    |> Code.eval_string()
    |> elem(0)
  end
end
