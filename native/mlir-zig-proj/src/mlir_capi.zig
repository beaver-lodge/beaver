const beam = @import("beam");
const kinda = @import("kinda");
pub const c = @import("prelude.zig");
pub const root_module = "Elixir.Beaver.MLIR.CAPI";
pub const Pass = kinda.ResourceKind(c.MlirPass, "Elixir.Beaver.MLIR.Pass");
pub const LogicalResult = kinda.ResourceKind(c.MlirLogicalResult, "Elixir.Beaver.MLIR.LogicalResult");
pub const StringRef = kinda.ResourceKind(c.MlirStringRef, "Elixir.Beaver.MLIR.StringRef");
pub const Context = kinda.ResourceKind(c.MlirContext, "Elixir.Beaver.MLIR.Context");
pub const Location = kinda.ResourceKind(c.MlirLocation, "Elixir.Beaver.MLIR.Location");
pub const ISize = kinda.ResourceKind(isize, "Elixir.Beaver.Native.ISize");
pub const Attribute = kinda.ResourceKind(c.MlirAttribute, "Elixir.Beaver.MLIR.Attribute");
pub const OpaquePtr = kinda.ResourceKind(?*anyopaque, "Elixir.Beaver.Native.OpaquePtr");
pub const MlirShapedTypeComponentsCallback = kinda.ResourceKind(c.MlirShapedTypeComponentsCallback, "Elixir.Beaver.MLIR.CAPI.MlirShapedTypeComponentsCallback");
pub const MlirTypeID = kinda.ResourceKind(c.MlirTypeID, "Elixir.Beaver.MLIR.CAPI.MlirTypeID");
pub const MlirTypesCallback = kinda.ResourceKind(c.MlirTypesCallback, "Elixir.Beaver.MLIR.CAPI.MlirTypesCallback");
pub const Bool = kinda.ResourceKind(bool, "Elixir.Beaver.Native.Bool");
pub const Operation = kinda.ResourceKind(c.MlirOperation, "Elixir.Beaver.MLIR.Operation");
pub const MlirIntegerSet = kinda.ResourceKind(c.MlirIntegerSet, "Elixir.Beaver.MLIR.CAPI.MlirIntegerSet");
pub const MlirAffineExpr = kinda.ResourceKind(c.MlirAffineExpr, "Elixir.Beaver.MLIR.CAPI.MlirAffineExpr");
pub const MlirStringCallback = kinda.ResourceKind(c.MlirStringCallback, "Elixir.Beaver.MLIR.CAPI.MlirStringCallback");
pub const MlirDialectHandle = kinda.ResourceKind(c.MlirDialectHandle, "Elixir.Beaver.MLIR.CAPI.MlirDialectHandle");
pub const CInt = kinda.ResourceKind(c_int, "Elixir.Beaver.Native.CInt");
pub const AffineMap = kinda.ResourceKind(c.MlirAffineMap, "Elixir.Beaver.MLIR.AffineMap");
pub const Enum_MlirSparseTensorLevelType = kinda.ResourceKind(c.enum_MlirSparseTensorLevelType, "Elixir.Beaver.MLIR.CAPI.Enum_MlirSparseTensorLevelType");
pub const F64 = kinda.ResourceKind(f64, "Elixir.Beaver.Native.F64");
pub const Type = kinda.ResourceKind(c.MlirType, "Elixir.Beaver.MLIR.Type");
pub const I32 = kinda.ResourceKind(i32, "Elixir.Beaver.Native.I32");
pub const I64 = kinda.ResourceKind(i64, "Elixir.Beaver.Native.I64");
pub const CUInt = kinda.ResourceKind(c_uint, "Elixir.Beaver.Native.CUInt");
pub const MlirDialectRegistry = kinda.ResourceKind(c.MlirDialectRegistry, "Elixir.Beaver.MLIR.CAPI.MlirDialectRegistry");
pub const MlirDiagnosticHandlerID = kinda.ResourceKind(c.MlirDiagnosticHandlerID, "Elixir.Beaver.MLIR.CAPI.MlirDiagnosticHandlerID");
pub const MlirDiagnosticHandler = kinda.ResourceKind(c.MlirDiagnosticHandler, "Elixir.Beaver.MLIR.CAPI.MlirDiagnosticHandler");
pub const DiagnosticHandlerDeleteUserData = kinda.ResourceKind(c.DiagnosticHandlerDeleteUserData, "Elixir.Beaver.MLIR.DiagnosticHandlerDeleteUserData");
pub const MlirDiagnostic = kinda.ResourceKind(c.MlirDiagnostic, "Elixir.Beaver.MLIR.CAPI.MlirDiagnostic");
pub const MlirDiagnosticSeverity = kinda.ResourceKind(c.MlirDiagnosticSeverity, "Elixir.Beaver.MLIR.CAPI.MlirDiagnosticSeverity");
pub const F32 = kinda.ResourceKind(f32, "Elixir.Beaver.Native.F32");
pub const U64 = kinda.ResourceKind(u64, "Elixir.Beaver.Native.U64");
pub const U32 = kinda.ResourceKind(u32, "Elixir.Beaver.Native.U32");
pub const U16 = kinda.ResourceKind(u16, "Elixir.Beaver.Native.U16");
pub const I16 = kinda.ResourceKind(i16, "Elixir.Beaver.Native.I16");
pub const U8 = kinda.ResourceKind(u8, "Elixir.Beaver.Native.U8");
pub const I8 = kinda.ResourceKind(i8, "Elixir.Beaver.Native.I8");
pub const USize = kinda.ResourceKind(usize, "Elixir.Beaver.Native.USize");
pub const UnmanagedDenseResourceElementsAttrGetDeleteCallback = kinda.ResourceKind(c.UnmanagedDenseResourceElementsAttrGetDeleteCallback, "Elixir.Beaver.MLIR.UnmanagedDenseResourceElementsAttrGetDeleteCallback");
pub const OpaqueArray = kinda.ResourceKind(?*const anyopaque, "Elixir.Beaver.Native.OpaqueArray");
pub const NamedAttribute = kinda.ResourceKind(c.MlirNamedAttribute, "Elixir.Beaver.MLIR.NamedAttribute");
pub const MlirPassManager = kinda.ResourceKind(c.MlirPassManager, "Elixir.Beaver.MLIR.CAPI.MlirPassManager");
pub const MlirRewritePatternSet = kinda.ResourceKind(c.MlirRewritePatternSet, "Elixir.Beaver.MLIR.CAPI.MlirRewritePatternSet");
pub const Region = kinda.ResourceKind(c.MlirRegion, "Elixir.Beaver.MLIR.Region");
pub const Module = kinda.ResourceKind(c.MlirModule, "Elixir.Beaver.MLIR.Module");
pub const MlirExecutionEngine = kinda.ResourceKind(c.MlirExecutionEngine, "Elixir.Beaver.MLIR.CAPI.MlirExecutionEngine");
pub const GenericCallback = kinda.ResourceKind(c.GenericCallback, "Elixir.Beaver.MLIR.GenericCallback");
pub const ExternalPassConstruct = kinda.ResourceKind(c.ExternalPassConstruct, "Elixir.Beaver.MLIR.ExternalPassConstruct");
pub const ExternalPassRun = kinda.ResourceKind(c.ExternalPassRun, "Elixir.Beaver.MLIR.ExternalPassRun");
pub const Identifier = kinda.ResourceKind(c.MlirIdentifier, "Elixir.Beaver.MLIR.Identifier");
pub const MlirOperationState = kinda.ResourceKind(c.MlirOperationState, "Elixir.Beaver.MLIR.CAPI.MlirOperationState");
pub const MlirSymbolTable = kinda.ResourceKind(c.MlirSymbolTable, "Elixir.Beaver.MLIR.CAPI.MlirSymbolTable");
pub const Value = kinda.ResourceKind(c.MlirValue, "Elixir.Beaver.MLIR.Value");
pub const Block = kinda.ResourceKind(c.MlirBlock, "Elixir.Beaver.MLIR.Block");
pub const Dialect = kinda.ResourceKind(c.MlirDialect, "Elixir.Beaver.MLIR.Dialect");
pub const MlirRegisteredOperationName = kinda.ResourceKind(c.MlirRegisteredOperationName, "Elixir.Beaver.MLIR.CAPI.MlirRegisteredOperationName");
pub const MlirExternalPass = kinda.ResourceKind(c.MlirExternalPass, "Elixir.Beaver.MLIR.CAPI.MlirExternalPass");
pub const MlirExternalPassCallbacks = kinda.ResourceKind(c.MlirExternalPassCallbacks, "Elixir.Beaver.MLIR.CAPI.MlirExternalPassCallbacks");
pub const MlirOpPassManager = kinda.ResourceKind(c.MlirOpPassManager, "Elixir.Beaver.MLIR.CAPI.MlirOpPassManager");
pub const AffineMapCompressUnusedSymbolsPopulateResult = kinda.ResourceKind(c.AffineMapCompressUnusedSymbolsPopulateResult, "Elixir.Beaver.MLIR.AffineMapCompressUnusedSymbolsPopulateResult");
pub const Struct_MlirAffineMap = kinda.ResourceKind(c.struct_MlirAffineMap, "Elixir.Beaver.MLIR.CAPI.Struct_MlirAffineMap");
pub const SymbolTableWalkSymbolTablesCallback = kinda.ResourceKind(c.SymbolTableWalkSymbolTablesCallback, "Elixir.Beaver.MLIR.SymbolTableWalkSymbolTablesCallback");
pub const OpOperand = kinda.ResourceKind(c.MlirOpOperand, "Elixir.Beaver.MLIR.OpOperand");
pub const MlirAsmState = kinda.ResourceKind(c.MlirAsmState, "Elixir.Beaver.MLIR.CAPI.MlirAsmState");
pub const MlirOperationWalkCallback = kinda.ResourceKind(c.MlirOperationWalkCallback, "Elixir.Beaver.MLIR.CAPI.MlirOperationWalkCallback");
pub const MlirWalkOrder = kinda.ResourceKind(c.MlirWalkOrder, "Elixir.Beaver.MLIR.CAPI.MlirWalkOrder");
pub const MlirBytecodeWriterConfig = kinda.ResourceKind(c.MlirBytecodeWriterConfig, "Elixir.Beaver.MLIR.CAPI.MlirBytecodeWriterConfig");
pub const MlirOpPrintingFlags = kinda.ResourceKind(c.MlirOpPrintingFlags, "Elixir.Beaver.MLIR.CAPI.MlirOpPrintingFlags");
pub const MlirLlvmThreadPool = kinda.ResourceKind(c.MlirLlvmThreadPool, "Elixir.Beaver.MLIR.CAPI.MlirLlvmThreadPool");
pub const MlirTypeIDAllocator = kinda.ResourceKind(c.MlirTypeIDAllocator, "Elixir.Beaver.MLIR.CAPI.MlirTypeIDAllocator");

pub const allKinds = .{
    Pass,
    LogicalResult,
    StringRef,
    Context,
    Location,
    ISize,
    Attribute,
    OpaquePtr,
    MlirShapedTypeComponentsCallback,
    MlirTypeID,
    MlirTypesCallback,
    Bool,
    Operation,
    MlirIntegerSet,
    MlirAffineExpr,
    MlirStringCallback,
    MlirDialectHandle,
    CInt,
    AffineMap,
    Enum_MlirSparseTensorLevelType,
    F64,
    Type,
    I32,
    I64,
    CUInt,
    MlirDialectRegistry,
    MlirDiagnosticHandlerID,
    MlirDiagnosticHandler,
    DiagnosticHandlerDeleteUserData,
    MlirDiagnostic,
    MlirDiagnosticSeverity,
    F32,
    U64,
    U32,
    U16,
    I16,
    U8,
    I8,
    USize,
    UnmanagedDenseResourceElementsAttrGetDeleteCallback,
    OpaqueArray,
    NamedAttribute,
    MlirPassManager,
    MlirRewritePatternSet,
    Region,
    Module,
    MlirExecutionEngine,
    GenericCallback,
    ExternalPassConstruct,
    ExternalPassRun,
    Identifier,
    MlirOperationState,
    MlirSymbolTable,
    Value,
    Block,
    Dialect,
    MlirRegisteredOperationName,
    MlirExternalPass,
    MlirExternalPassCallbacks,
    MlirOpPassManager,
    AffineMapCompressUnusedSymbolsPopulateResult,
    Struct_MlirAffineMap,
    SymbolTableWalkSymbolTablesCallback,
    OpOperand,
    MlirAsmState,
    MlirOperationWalkCallback,
    MlirWalkOrder,
    MlirBytecodeWriterConfig,
    MlirOpPrintingFlags,
    MlirLlvmThreadPool,
    MlirTypeIDAllocator,
};
pub fn open_generated_resource_types(env: beam.env) void {
    inline for (allKinds) |k| {
        k.open_all(env);
    }
    kinda.aliasKind(OpaquePtr, kinda.Internal.OpaquePtr);
    kinda.aliasKind(OpaqueArray, kinda.Internal.OpaqueArray);
    kinda.aliasKind(USize, kinda.Internal.USize);
}
