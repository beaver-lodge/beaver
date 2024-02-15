const beam = @import("beam");
const kinda = @import("kinda");
pub const c = @import("prelude.zig");
pub const root_module = "Elixir.Beaver.MLIR.CAPI";
fn NativeKind(comptime t: type, comptime n: [*]const u8) type {
    const nativeModPrefix = "Elixir.Beaver.Native.";
    return kinda.ResourceKind(t, nativeModPrefix ++ n);
}
pub const ISize = NativeKind(isize, "ISize");
pub const OpaquePtr = NativeKind(?*anyopaque, "OpaquePtr");
pub const Bool = NativeKind(bool, "Bool");
pub const CInt = NativeKind(c_int, "CInt");
pub const F64 = NativeKind(f64, "F64");
pub const I32 = NativeKind(i32, "I32");
pub const I64 = NativeKind(i64, "I64");
pub const CUInt = NativeKind(c_uint, "CUInt");
pub const F32 = NativeKind(f32, "F32");
pub const U64 = NativeKind(u64, "U64");
pub const U32 = NativeKind(u32, "U32");
pub const U16 = NativeKind(u16, "U16");
pub const I8 = NativeKind(i8, "I8");
pub const I16 = NativeKind(i16, "I16");
pub const U8 = NativeKind(u8, "U8");
pub const USize = NativeKind(usize, "USize");
pub const OpaqueArray = NativeKind(?*const anyopaque, "OpaqueArray");

fn MLIRKind(comptime t: type, comptime n: [*]const u8) type {
    const nsPrefix = "Elixir.Beaver.MLIR.";
    return kinda.ResourceKind(t, nsPrefix ++ n);
}
pub const Type = MLIRKind(c.MlirType, "Type");
pub const Pass = MLIRKind(c.MlirPass, "Pass");
pub const LogicalResult = MLIRKind(c.MlirLogicalResult, "LogicalResult");
pub const StringRef = MLIRKind(c.MlirStringRef, "StringRef");
pub const Context = MLIRKind(c.MlirContext, "Context");
pub const Location = MLIRKind(c.MlirLocation, "Location");
pub const Attribute = MLIRKind(c.MlirAttribute, "Attribute");
pub const MlirShapedTypeComponentsCallback = MLIRKind(c.MlirShapedTypeComponentsCallback, "CAPI.MlirShapedTypeComponentsCallback");
pub const MlirTypeID = MLIRKind(c.MlirTypeID, "CAPI.MlirTypeID");
pub const MlirTypesCallback = MLIRKind(c.MlirTypesCallback, "CAPI.MlirTypesCallback");
pub const Operation = MLIRKind(c.MlirOperation, "Operation");
pub const MlirIntegerSet = MLIRKind(c.MlirIntegerSet, "CAPI.MlirIntegerSet");
pub const MlirAffineExpr = MLIRKind(c.MlirAffineExpr, "CAPI.MlirAffineExpr");
pub const MlirStringCallback = MLIRKind(c.MlirStringCallback, "CAPI.MlirStringCallback");
pub const MlirDialectHandle = MLIRKind(c.MlirDialectHandle, "CAPI.MlirDialectHandle");
pub const AffineMap = MLIRKind(c.MlirAffineMap, "AffineMap");
pub const Enum_MlirSparseTensorLevelType = MLIRKind(c.enum_MlirSparseTensorLevelType, "CAPI.Enum_MlirSparseTensorLevelType");
pub const MlirDialectRegistry = MLIRKind(c.MlirDialectRegistry, "CAPI.MlirDialectRegistry");
pub const MlirDiagnosticHandlerID = MLIRKind(c.MlirDiagnosticHandlerID, "CAPI.MlirDiagnosticHandlerID");
pub const MlirDiagnosticHandler = MLIRKind(c.MlirDiagnosticHandler, "CAPI.MlirDiagnosticHandler");
pub const DiagnosticHandlerDeleteUserData = MLIRKind(c.DiagnosticHandlerDeleteUserData, "DiagnosticHandlerDeleteUserData");
pub const MlirDiagnostic = MLIRKind(c.MlirDiagnostic, "CAPI.MlirDiagnostic");
pub const MlirDiagnosticSeverity = MLIRKind(c.MlirDiagnosticSeverity, "CAPI.MlirDiagnosticSeverity");
pub const UnmanagedDenseResourceElementsAttrGetDeleteCallback = MLIRKind(c.UnmanagedDenseResourceElementsAttrGetDeleteCallback, "UnmanagedDenseResourceElementsAttrGetDeleteCallback");
pub const NamedAttribute = MLIRKind(c.MlirNamedAttribute, "NamedAttribute");
pub const MlirPassManager = MLIRKind(c.MlirPassManager, "CAPI.MlirPassManager");
pub const MlirRewritePatternSet = MLIRKind(c.MlirRewritePatternSet, "CAPI.MlirRewritePatternSet");
pub const Region = MLIRKind(c.MlirRegion, "Region");
pub const Module = MLIRKind(c.MlirModule, "Module");
pub const MlirExecutionEngine = MLIRKind(c.MlirExecutionEngine, "CAPI.MlirExecutionEngine");
pub const GenericCallback = MLIRKind(c.GenericCallback, "GenericCallback");
pub const ExternalPassConstruct = MLIRKind(c.ExternalPassConstruct, "ExternalPassConstruct");
pub const ExternalPassRun = MLIRKind(c.ExternalPassRun, "ExternalPassRun");
pub const Identifier = MLIRKind(c.MlirIdentifier, "Identifier");
pub const MlirOperationState = MLIRKind(c.MlirOperationState, "CAPI.MlirOperationState");
pub const MlirSymbolTable = MLIRKind(c.MlirSymbolTable, "CAPI.MlirSymbolTable");
pub const Value = MLIRKind(c.MlirValue, "Value");
pub const Block = MLIRKind(c.MlirBlock, "Block");
pub const Dialect = MLIRKind(c.MlirDialect, "Dialect");
pub const MlirRegisteredOperationName = MLIRKind(c.MlirRegisteredOperationName, "CAPI.MlirRegisteredOperationName");
pub const MlirExternalPass = MLIRKind(c.MlirExternalPass, "CAPI.MlirExternalPass");
pub const MlirExternalPassCallbacks = MLIRKind(c.MlirExternalPassCallbacks, "CAPI.MlirExternalPassCallbacks");
pub const MlirOpPassManager = MLIRKind(c.MlirOpPassManager, "CAPI.MlirOpPassManager");
pub const AffineMapCompressUnusedSymbolsPopulateResult = MLIRKind(c.AffineMapCompressUnusedSymbolsPopulateResult, "AffineMapCompressUnusedSymbolsPopulateResult");
pub const Struct_MlirAffineMap = MLIRKind(c.struct_MlirAffineMap, "CAPI.Struct_MlirAffineMap");
pub const SymbolTableWalkSymbolTablesCallback = MLIRKind(c.SymbolTableWalkSymbolTablesCallback, "SymbolTableWalkSymbolTablesCallback");
pub const OpOperand = MLIRKind(c.MlirOpOperand, "OpOperand");
pub const MlirAsmState = MLIRKind(c.MlirAsmState, "CAPI.MlirAsmState");
pub const MlirOperationWalkCallback = MLIRKind(c.MlirOperationWalkCallback, "CAPI.MlirOperationWalkCallback");
pub const MlirWalkOrder = MLIRKind(c.MlirWalkOrder, "CAPI.MlirWalkOrder");
pub const MlirBytecodeWriterConfig = MLIRKind(c.MlirBytecodeWriterConfig, "CAPI.MlirBytecodeWriterConfig");
pub const MlirOpPrintingFlags = MLIRKind(c.MlirOpPrintingFlags, "CAPI.MlirOpPrintingFlags");
pub const MlirLlvmThreadPool = MLIRKind(c.MlirLlvmThreadPool, "CAPI.MlirLlvmThreadPool");
pub const MlirTypeIDAllocator = MLIRKind(c.MlirTypeIDAllocator, "CAPI.MlirTypeIDAllocator");

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
