defmodule Beaver.MLIR.CAPI.CodeGen do
  @moduledoc false
  alias Kinda.CodeGen.{KindDecl}
  @behaviour Kinda.CodeGen

  @impl Kinda.CodeGen
  def kinds() do
    Enum.map(
      ~w[
          Type
          Pass
          LogicalResult
          StringRef
          Context
          Location
          Attribute
          Operation
          AffineMap
          DiagnosticHandlerDeleteUserData
          NamedAttribute
          Region
          Module
          GenericCallback
          ExternalPassConstruct
          ExternalPassRun
          Identifier
          Value
          Block
          Dialect
          SymbolTableWalkSymbolTablesCallback
          OpOperand
          AffineMapCompressUnusedSymbolsPopulateResult
          UnmanagedDenseResourceElementsAttrGetDeleteCallback
          SparseTensorLevelType
          ShapedTypeComponentsCallback
          TypeID
          TypesCallback
          IntegerSet
          AffineExpr
          StringCallback
          DialectHandle
          DialectRegistry
          DiagnosticHandlerID
          DiagnosticHandler
          Diagnostic
          DiagnosticSeverity
          PassManager
          RewritePatternSet
          ExecutionEngine
          Operation.State
          SymbolTable
          ExternalPass
          ExternalPassCallbacks
          OpPassManager
          AsmState
          OperationWalkCallback
          WalkOrder
          BytecodeWriterConfig
          OpPrintingFlags
          LLVMThreadPool
          TypeIDAllocator
          RewriterBase
          FrozenRewritePatternSet
          PDLPatternModule
          GreedyRewriteDriverConfig
          TransformOptions
          LinalgContractionDimensions
          LinalgConvolutionDimensions
          PDLValue
          PDLResultList
          PDLRewriteFunction
          PatternRewriter
          UnrankedMemRefDescriptor
        ],
      &%KindDecl{module_name: Module.concat(Beaver.MLIR, &1)}
    ) ++
      Enum.map(
        ~w[
          ISize
          OpaquePtr
          Bool
          CInt
          F64
          I32
          I64
          CUInt
          F32
          U64
          U32
          U16
          I8
          I16
          U8
          USize
          OpaqueArray
          StringArray
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
