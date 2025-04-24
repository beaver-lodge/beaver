const beam = @import("beam");
const kinda = @import("kinda");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const string_ref = @import("string_ref.zig");
pub const root_module = "Elixir.Beaver.MLIR.CAPI";
fn NativeKind(comptime t: type, comptime n: []const u8) type {
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
pub const StringArray = NativeKind([*c][*c]const u8, "StringArray");

fn MLIRKind(comptime n: []const u8) type {
    const nsPrefix = "Elixir.Beaver.MLIR.";
    return kinda.ResourceKind(@field(c, "Mlir" ++ n), nsPrefix ++ n);
}
fn MLIRKind2(comptime s: []const u8, comptime n: []const u8) type {
    const nsPrefix = "Elixir.Beaver.MLIR.";
    return kinda.ResourceKind(@field(c, s), nsPrefix ++ n);
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
pub const SparseTensorLevelType = MLIRKind("SparseTensorLevelType");
pub const ShapedTypeComponentsCallback = MLIRKind("ShapedTypeComponentsCallback");
pub const TypeID = MLIRKind("TypeID");
pub const TypesCallback = MLIRKind("TypesCallback");
pub const IntegerSet = MLIRKind("IntegerSet");
pub const AffineExpr = MLIRKind("AffineExpr");
pub const StringCallback = MLIRKind("StringCallback");
pub const DialectHandle = MLIRKind("DialectHandle");
pub const DialectRegistry = MLIRKind("DialectRegistry");
pub const DiagnosticHandlerID = MLIRKind("DiagnosticHandlerID");
pub const DiagnosticHandler = MLIRKind("DiagnosticHandler");
pub const Diagnostic = MLIRKind("Diagnostic");
pub const DiagnosticSeverity = MLIRKind("DiagnosticSeverity");
pub const PassManager = MLIRKind("PassManager");
pub const RewritePatternSet = MLIRKind("RewritePatternSet");
pub const ExecutionEngine = MLIRKind("ExecutionEngine");
pub const SymbolTable = MLIRKind("SymbolTable");
pub const RewriterBase = MLIRKind("RewriterBase");
pub const FrozenRewritePatternSet = MLIRKind("FrozenRewritePatternSet");
pub const PDLPatternModule = MLIRKind("PDLPatternModule");
pub const GreedyRewriteDriverConfig = MLIRKind("GreedyRewriteDriverConfig");
pub const LinalgContractionDimensions = MLIRKind("LinalgContractionDimensions");
pub const LinalgConvolutionDimensions = MLIRKind("LinalgConvolutionDimensions");

pub const ExternalPass = MLIRKind("ExternalPass");
pub const ExternalPassCallbacks = MLIRKind("ExternalPassCallbacks");
pub const OpPassManager = MLIRKind("OpPassManager");
pub const AsmState = MLIRKind("AsmState");
pub const OperationWalkCallback = MLIRKind("OperationWalkCallback");
pub const WalkOrder = MLIRKind("WalkOrder");
pub const BytecodeWriterConfig = MLIRKind("BytecodeWriterConfig");
pub const OpPrintingFlags = MLIRKind("OpPrintingFlags");
pub const TypeIDAllocator = MLIRKind("TypeIDAllocator");
pub const TransformOptions = MLIRKind("TransformOptions");
pub const LLVMThreadPool = MLIRKind2("MlirLlvmThreadPool", "LLVMThreadPool");
pub const OperationState = MLIRKind2("MlirOperationState", "Operation.State");
pub const allKinds = .{ Pass, LogicalResult, StringRef, Context, Location, ISize, Attribute, OpaquePtr, ShapedTypeComponentsCallback, TypeID, TypesCallback, Bool, Operation, IntegerSet, AffineExpr, StringCallback, DialectHandle, CInt, AffineMap, SparseTensorLevelType, F64, Type, I32, I64, CUInt, DialectRegistry, DiagnosticHandlerID, DiagnosticHandler, DiagnosticHandlerDeleteUserData, Diagnostic, DiagnosticSeverity, F32, U64, U32, U16, I16, U8, I8, USize, UnmanagedDenseResourceElementsAttrGetDeleteCallback, OpaqueArray, StringArray, NamedAttribute, PassManager, RewritePatternSet, Region, Module, ExecutionEngine, GenericCallback, ExternalPassConstruct, ExternalPassRun, Identifier, OperationState, SymbolTable, Value, Block, Dialect, ExternalPass, ExternalPassCallbacks, OpPassManager, AffineMapCompressUnusedSymbolsPopulateResult, SymbolTableWalkSymbolTablesCallback, OpOperand, AsmState, OperationWalkCallback, WalkOrder, BytecodeWriterConfig, OpPrintingFlags, LLVMThreadPool, TypeIDAllocator, TransformOptions, RewriterBase, FrozenRewritePatternSet, PDLPatternModule, GreedyRewriteDriverConfig, string_ref.Printer.ResourceKind, LinalgContractionDimensions, LinalgConvolutionDimensions };
pub fn open_all(env: beam.env) void {
    inline for (allKinds) |k| {
        k.open_all(env);
    }
    kinda.aliasKind(OpaquePtr, kinda.Internal.OpaquePtr);
    kinda.aliasKind(OpaqueArray, kinda.Internal.OpaqueArray);
    kinda.aliasKind(USize, kinda.Internal.USize);
    kinda.aliasKind(DiagnosticHandlerID, U64);
    kinda.aliasKind(SparseTensorLevelType, U64);
}

const EntriesT = [allKinds.len * kinda.numOfNIFsPerKind]e.ErlNifFunc;
pub const EntriesOfKinds = getEntries();
fn getEntries() EntriesT {
    var ret: EntriesT = undefined;
    @setEvalBranchQuota(8000);
    for (allKinds, 0..) |k, i| {
        for (0..kinda.numOfNIFsPerKind) |j| {
            ret[i * kinda.numOfNIFsPerKind + j] = k.nifs[j];
        }
    }
    return ret;
}
