const beam = @import("beam");
const kinda = @import("kinda");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
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

fn MLIRKind(comptime n: [*]const u8) type {
    const nsPrefix = "Elixir.Beaver.MLIR.";
    return kinda.ResourceKind(@field(c, "Mlir" ++ n), nsPrefix ++ n);
}
pub const Type = MLIRKind("Type");
pub const Pass = MLIRKind("Pass");
pub const LogicalResult = MLIRKind("LogicalResult");
pub const StringRef = MLIRKind("StringRef");
pub const Context = MLIRKind("Context");
pub const Location = MLIRKind("Location");
pub const Attribute = MLIRKind("Attribute");
pub const Operation = MLIRKind("Operation");
pub const AffineMap = MLIRKind("AffineMap");
pub const DiagnosticHandlerDeleteUserData = MLIRKind("DiagnosticHandlerDeleteUserData");
pub const NamedAttribute = MLIRKind("NamedAttribute");
pub const Region = MLIRKind("Region");
pub const Module = MLIRKind("Module");
pub const GenericCallback = MLIRKind("GenericCallback");
pub const ExternalPassConstruct = MLIRKind("ExternalPassConstruct");
pub const ExternalPassRun = MLIRKind("ExternalPassRun");
pub const Identifier = MLIRKind("Identifier");
pub const Value = MLIRKind("Value");
pub const Block = MLIRKind("Block");
pub const Dialect = MLIRKind("Dialect");
pub const SymbolTableWalkSymbolTablesCallback = MLIRKind("SymbolTableWalkSymbolTablesCallback");
pub const OpOperand = MLIRKind("OpOperand");
pub const AffineMapCompressUnusedSymbolsPopulateResult = MLIRKind("AffineMapCompressUnusedSymbolsPopulateResult");
pub const UnmanagedDenseResourceElementsAttrGetDeleteCallback = MLIRKind("UnmanagedDenseResourceElementsAttrGetDeleteCallback");

pub const Enum_MlirSparseTensorLevelType = kinda.ResourceKind(c.MlirSparseTensorLevelType, root_module ++ "." ++ "Enum_MlirSparseTensorLevelType");

fn MLIRCAPIKind(comptime n: []const u8) type {
    return kinda.ResourceKind(@field(c, n), root_module ++ "." ++ n);
}

pub const MlirShapedTypeComponentsCallback = MLIRCAPIKind("MlirShapedTypeComponentsCallback");
pub const MlirTypeID = MLIRCAPIKind("MlirTypeID");
pub const MlirTypesCallback = MLIRCAPIKind("MlirTypesCallback");
pub const MlirIntegerSet = MLIRCAPIKind("MlirIntegerSet");
pub const MlirAffineExpr = MLIRCAPIKind("MlirAffineExpr");
pub const MlirStringCallback = MLIRCAPIKind("MlirStringCallback");
pub const MlirDialectHandle = MLIRCAPIKind("MlirDialectHandle");
pub const MlirDialectRegistry = MLIRCAPIKind("MlirDialectRegistry");
pub const MlirDiagnosticHandlerID = MLIRCAPIKind("MlirDiagnosticHandlerID");
pub const MlirDiagnosticHandler = MLIRCAPIKind("MlirDiagnosticHandler");
pub const MlirDiagnostic = MLIRCAPIKind("MlirDiagnostic");
pub const MlirDiagnosticSeverity = MLIRCAPIKind("MlirDiagnosticSeverity");
pub const MlirPassManager = MLIRCAPIKind("MlirPassManager");
pub const MlirRewritePatternSet = MLIRCAPIKind("MlirRewritePatternSet");
pub const MlirExecutionEngine = MLIRCAPIKind("MlirExecutionEngine");
pub const MlirOperationState = MLIRCAPIKind("MlirOperationState");
pub const MlirSymbolTable = MLIRCAPIKind("MlirSymbolTable");
pub const MlirRegisteredOperationName = MLIRCAPIKind("MlirRegisteredOperationName");
pub const MlirExternalPass = MLIRCAPIKind("MlirExternalPass");
pub const MlirExternalPassCallbacks = MLIRCAPIKind("MlirExternalPassCallbacks");
pub const MlirOpPassManager = MLIRCAPIKind("MlirOpPassManager");
pub const MlirAsmState = MLIRCAPIKind("MlirAsmState");
pub const MlirOperationWalkCallback = MLIRCAPIKind("MlirOperationWalkCallback");
pub const MlirWalkOrder = MLIRCAPIKind("MlirWalkOrder");
pub const MlirBytecodeWriterConfig = MLIRCAPIKind("MlirBytecodeWriterConfig");
pub const MlirOpPrintingFlags = MLIRCAPIKind("MlirOpPrintingFlags");
pub const MlirLlvmThreadPool = MLIRCAPIKind("MlirLlvmThreadPool");
pub const MlirTypeIDAllocator = MLIRCAPIKind("MlirTypeIDAllocator");

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

const numOfNIFsPerKind = 10;
const EntriesT = [allKinds.len * numOfNIFsPerKind]e.ErlNifFunc;
pub const EntriesOfKinds = getEntries();
fn getEntries() EntriesT {
    var ret: EntriesT = undefined;
    @setEvalBranchQuota(8000);
    const Kinds = allKinds;
    for (Kinds, 0..) |k, i| {
        for (0..numOfNIFsPerKind) |j| {
            ret[i * numOfNIFsPerKind + j] = k.nifs[j];
        }
    }
    return ret;
}
