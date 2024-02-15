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
pub const Operation = MLIRKind(c.MlirOperation, "Operation");
pub const AffineMap = MLIRKind(c.MlirAffineMap, "AffineMap");
pub const DiagnosticHandlerDeleteUserData = MLIRKind(c.DiagnosticHandlerDeleteUserData, "DiagnosticHandlerDeleteUserData");
pub const NamedAttribute = MLIRKind(c.MlirNamedAttribute, "NamedAttribute");
pub const Region = MLIRKind(c.MlirRegion, "Region");
pub const Module = MLIRKind(c.MlirModule, "Module");
pub const GenericCallback = MLIRKind(c.GenericCallback, "GenericCallback");
pub const ExternalPassConstruct = MLIRKind(c.ExternalPassConstruct, "ExternalPassConstruct");
pub const ExternalPassRun = MLIRKind(c.ExternalPassRun, "ExternalPassRun");
pub const Identifier = MLIRKind(c.MlirIdentifier, "Identifier");
pub const Value = MLIRKind(c.MlirValue, "Value");
pub const Block = MLIRKind(c.MlirBlock, "Block");
pub const Dialect = MLIRKind(c.MlirDialect, "Dialect");
pub const SymbolTableWalkSymbolTablesCallback = MLIRKind(c.SymbolTableWalkSymbolTablesCallback, "SymbolTableWalkSymbolTablesCallback");
pub const OpOperand = MLIRKind(c.MlirOpOperand, "OpOperand");
pub const AffineMapCompressUnusedSymbolsPopulateResult = MLIRKind(c.AffineMapCompressUnusedSymbolsPopulateResult, "AffineMapCompressUnusedSymbolsPopulateResult");
pub const UnmanagedDenseResourceElementsAttrGetDeleteCallback = MLIRKind(c.UnmanagedDenseResourceElementsAttrGetDeleteCallback, "UnmanagedDenseResourceElementsAttrGetDeleteCallback");

fn MLIRCAPIKind(comptime t: type, comptime n: [*]const u8) type {
    return kinda.ResourceKind(t, root_module ++ "." ++ n);
}

pub const MlirShapedTypeComponentsCallback = MLIRCAPIKind(c.MlirShapedTypeComponentsCallback, "MlirShapedTypeComponentsCallback");
pub const MlirTypeID = MLIRCAPIKind(c.MlirTypeID, "MlirTypeID");
pub const MlirTypesCallback = MLIRCAPIKind(c.MlirTypesCallback, "MlirTypesCallback");
pub const MlirIntegerSet = MLIRCAPIKind(c.MlirIntegerSet, "MlirIntegerSet");
pub const MlirAffineExpr = MLIRCAPIKind(c.MlirAffineExpr, "MlirAffineExpr");
pub const MlirStringCallback = MLIRCAPIKind(c.MlirStringCallback, "MlirStringCallback");
pub const MlirDialectHandle = MLIRCAPIKind(c.MlirDialectHandle, "MlirDialectHandle");
pub const Enum_MlirSparseTensorLevelType = MLIRCAPIKind(c.enum_MlirSparseTensorLevelType, "Enum_MlirSparseTensorLevelType");
pub const MlirDialectRegistry = MLIRCAPIKind(c.MlirDialectRegistry, "MlirDialectRegistry");
pub const MlirDiagnosticHandlerID = MLIRCAPIKind(c.MlirDiagnosticHandlerID, "MlirDiagnosticHandlerID");
pub const MlirDiagnosticHandler = MLIRCAPIKind(c.MlirDiagnosticHandler, "MlirDiagnosticHandler");
pub const MlirDiagnostic = MLIRCAPIKind(c.MlirDiagnostic, "MlirDiagnostic");
pub const MlirDiagnosticSeverity = MLIRCAPIKind(c.MlirDiagnosticSeverity, "MlirDiagnosticSeverity");
pub const MlirPassManager = MLIRCAPIKind(c.MlirPassManager, "MlirPassManager");
pub const MlirRewritePatternSet = MLIRCAPIKind(c.MlirRewritePatternSet, "MlirRewritePatternSet");
pub const MlirExecutionEngine = MLIRCAPIKind(c.MlirExecutionEngine, "MlirExecutionEngine");
pub const MlirOperationState = MLIRCAPIKind(c.MlirOperationState, "MlirOperationState");
pub const MlirSymbolTable = MLIRCAPIKind(c.MlirSymbolTable, "MlirSymbolTable");
pub const MlirRegisteredOperationName = MLIRCAPIKind(c.MlirRegisteredOperationName, "MlirRegisteredOperationName");
pub const MlirExternalPass = MLIRCAPIKind(c.MlirExternalPass, "MlirExternalPass");
pub const MlirExternalPassCallbacks = MLIRCAPIKind(c.MlirExternalPassCallbacks, "MlirExternalPassCallbacks");
pub const MlirOpPassManager = MLIRCAPIKind(c.MlirOpPassManager, "MlirOpPassManager");
pub const Struct_MlirAffineMap = MLIRCAPIKind(c.struct_MlirAffineMap, "Struct_MlirAffineMap");
pub const MlirAsmState = MLIRCAPIKind(c.MlirAsmState, "MlirAsmState");
pub const MlirOperationWalkCallback = MLIRCAPIKind(c.MlirOperationWalkCallback, "MlirOperationWalkCallback");
pub const MlirWalkOrder = MLIRCAPIKind(c.MlirWalkOrder, "MlirWalkOrder");
pub const MlirBytecodeWriterConfig = MLIRCAPIKind(c.MlirBytecodeWriterConfig, "MlirBytecodeWriterConfig");
pub const MlirOpPrintingFlags = MLIRCAPIKind(c.MlirOpPrintingFlags, "MlirOpPrintingFlags");
pub const MlirLlvmThreadPool = MLIRCAPIKind(c.MlirLlvmThreadPool, "MlirLlvmThreadPool");
pub const MlirTypeIDAllocator = MLIRCAPIKind(c.MlirTypeIDAllocator, "MlirTypeIDAllocator");

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
