pub const c = @import("prelude.zig").c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const string_ref = @import("string_ref.zig");
pub const root_module = "Elixir.Beaver.MLIR.CAPI.Raw";
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
pub const ElixirNameSpacePrefix = "Elixir.Beaver.MLIR.";

fn MLIRKind(comptime n: []const u8) type {
    return kinda.ResourceKind(@field(c, "Mlir" ++ n), ElixirNameSpacePrefix ++ n);
}
fn MLIRKind2(comptime s: []const u8, comptime n: []const u8) type {
    return kinda.ResourceKind(@field(c, s), ElixirNameSpacePrefix ++ n);
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
pub const RewritePattern = MLIRKind("RewritePattern");
pub const RewritePatternSet = MLIRKind("RewritePatternSet");
pub const RewritePatternCallbacks = MLIRKind("RewritePatternCallbacks");
pub const ConversionTarget = MLIRKind("ConversionTarget");
pub const ConversionPattern = MLIRKind("ConversionPattern");
pub const ConversionPatternRewriter = MLIRKind("ConversionPatternRewriter");
pub const TypeConverter = MLIRKind("TypeConverter");
pub const ConversionConfig = MLIRKind("ConversionConfig");
pub const ExecutionEngine = MLIRKind("ExecutionEngine");
pub const SymbolTable = MLIRKind("SymbolTable");
pub const RewriterBase = MLIRKind("RewriterBase");
pub const FrozenRewritePatternSet = MLIRKind("FrozenRewritePatternSet");
pub const PDLPatternModule = MLIRKind("PDLPatternModule");
pub const GreedyRewriteDriverConfig = MLIRKind("GreedyRewriteDriverConfig");
pub const LinalgContractionDimensions = MLIRKind("LinalgContractionDimensions");
pub const LinalgConvolutionDimensions = MLIRKind("LinalgConvolutionDimensions");
pub const PDLValue = MLIRKind("PDLValue");
pub const PDLResultList = MLIRKind("PDLResultList");
pub const PDLRewriteFunction = MLIRKind("PDLRewriteFunction");
pub const PatternRewriter = MLIRKind("PatternRewriter");
pub const ConditionallySpeculatableOpInterfaceCallbacks = MLIRKind("ConditionallySpeculatableOpInterfaceCallbacks");
pub const DominanceInfo = MLIRKind("DominanceInfo");
pub const IRMapping = MLIRKind("IRMapping");
pub const MemoryEffect = MLIRKind("MemoryEffect");
pub const MemoryEffectInstance = MLIRKind("MemoryEffectInstance");
pub const OpOperandReplaceFilterCallback = MLIRKind("OpOperandReplaceFilterCallback");
pub const PostDominanceInfo = MLIRKind("PostDominanceInfo");
pub const RewriterBaseInsertPoint = MLIRKind("RewriterBaseInsertPoint");
pub const SideEffectResource = MLIRKind("SideEffectResource");
pub const TypeConverter1ToNTargetMaterializationCallback = MLIRKind("TypeConverter1ToNTargetMaterializationCallback");
pub const TypeConverter1ToNConversionCallback = MLIRKind("TypeConverter1ToNConversionCallback");
pub const TypeConverterConversionResults = MLIRKind("TypeConverterConversionResults");
pub const TypeConverterSourceMaterializationCallback = MLIRKind("TypeConverterSourceMaterializationCallback");
pub const TypeConverterTargetMaterializationCallback = MLIRKind("TypeConverterTargetMaterializationCallback");

pub const ExternalPass = MLIRKind("ExternalPass");
pub const ExternalPassCallbacks = MLIRKind("ExternalPassCallbacks");
pub const OpPassManager = MLIRKind("OpPassManager");
pub const AsmState = MLIRKind("AsmState");
pub const OperationWalkCallback = MLIRKind("OperationWalkCallback");
pub const WalkOrder = MLIRKind("WalkOrder");
pub const BytecodeWriterConfig = MLIRKind("BytecodeWriterConfig");
pub const OpPrintingFlags = MLIRKind("OpPrintingFlags");
pub const LLVMRawFdOStream = MLIRKind2("MlirLlvmRawFdOStream", "LLVMRawFdOStream");
pub const TypeIDAllocator = MLIRKind("TypeIDAllocator");
pub const DynamicOpTrait = MLIRKind("DynamicOpTrait");
pub const DynamicOpTraitCallbacks = MLIRKind("DynamicOpTraitCallbacks");
pub const DynamicTypeDefinition = MLIRKind("DynamicTypeDefinition");
pub const DynamicAttrDefinition = MLIRKind("DynamicAttrDefinition");
pub const MemoryEffectInstancesList = MLIRKind2("MlirBeaverMemoryEffectInstancesList", "MemoryEffectInstancesList");
pub const TransformResults = MLIRKind("TransformResults");
pub const TransformRewriter = MLIRKind("TransformRewriter");
pub const TransformState = MLIRKind("TransformState");
pub const TransformOpInterfaceCallbacks = MLIRKind("TransformOpInterfaceCallbacks");
pub const PatternDescriptorOpInterfaceCallbacks = MLIRKind("PatternDescriptorOpInterfaceCallbacks");
pub const TransformOptions = MLIRKind("TransformOptions");
pub const LLVMThreadPool = MLIRKind2("MlirLlvmThreadPool", "LLVMThreadPool");
pub const OperationState = MLIRKind2("MlirOperationState", "Operation.State");
pub const allKinds = .{ Pass, LogicalResult, StringRef, Context, Location, ISize, Attribute, OpaquePtr, ShapedTypeComponentsCallback, TypeID, TypesCallback, Bool, Operation, IntegerSet, AffineExpr, StringCallback, DialectHandle, CInt, AffineMap, SparseTensorLevelType, F64, Type, I32, I64, CUInt, DialectRegistry, DiagnosticHandlerID, DiagnosticHandler, DiagnosticHandlerDeleteUserData, Diagnostic, DiagnosticSeverity, F32, U64, U32, U16, I16, U8, I8, USize, UnmanagedDenseResourceElementsAttrGetDeleteCallback, OpaqueArray, StringArray, NamedAttribute, PassManager, RewritePattern, RewritePatternSet, RewritePatternCallbacks, ConversionTarget, ConversionPattern, ConversionPatternRewriter, TypeConverter, ConversionConfig, Region, Module, ExecutionEngine, GenericCallback, ExternalPassConstruct, ExternalPassRun, Identifier, OperationState, SymbolTable, Value, Block, Dialect, ExternalPass, ExternalPassCallbacks, OpPassManager, AffineMapCompressUnusedSymbolsPopulateResult, SymbolTableWalkSymbolTablesCallback, OpOperand, AsmState, OperationWalkCallback, WalkOrder, BytecodeWriterConfig, OpPrintingFlags, LLVMRawFdOStream, LLVMThreadPool, TypeIDAllocator, DynamicOpTrait, DynamicOpTraitCallbacks, DynamicTypeDefinition, DynamicAttrDefinition, MemoryEffectInstancesList, TransformResults, TransformRewriter, TransformState, TransformOpInterfaceCallbacks, PatternDescriptorOpInterfaceCallbacks, TransformOptions, RewriterBase, FrozenRewritePatternSet, PDLPatternModule, GreedyRewriteDriverConfig, string_ref.Printer.ResourceKind, LinalgContractionDimensions, LinalgConvolutionDimensions, PDLValue, PDLResultList, PDLRewriteFunction, PatternRewriter, ConditionallySpeculatableOpInterfaceCallbacks, DominanceInfo, IRMapping, MemoryEffect, MemoryEffectInstance, OpOperandReplaceFilterCallback, PostDominanceInfo, RewriterBaseInsertPoint, SideEffectResource, TypeConverter1ToNConversionCallback, TypeConverter1ToNTargetMaterializationCallback, TypeConverterConversionResults, TypeConverterSourceMaterializationCallback, TypeConverterTargetMaterializationCallback };

/// Kinds that share ERTS resource types with another kind. The alias target
/// opens the slot; the alias kind only copies the handle so terms made through
/// either name fetch the same native data.
const kindAliases = .{
    .{ OpaquePtr, kinda.Internal.OpaquePtr },
    .{ OpaqueArray, kinda.Internal.OpaqueArray },
    .{ USize, kinda.Internal.USize },
    .{ DiagnosticHandlerID, U64 },
    .{ SparseTensorLevelType, U64 },
};

/// Every resource slot opened by this module, in a stable name order, plus
/// the shared internal kinds and the callback reply token. The core partition
/// publishes these handles by name for the leaf partitions; each partition
/// builds its own registry instance because comptime values do not cross DSO
/// boundaries, but the slot names and order are identical.
pub const resourceSlots = blk: {
    const internal_kinds = .{ kinda.Internal.OpaquePtr, kinda.Internal.OpaqueArray, kinda.Internal.USize, kinda.Internal.OpaqueStruct };
    var tuple: []const kinda.ResourceSlot = &.{};
    for (allKinds) |k| {
        tuple = tuple ++ &k.slots;
    }
    for (internal_kinds) |k| {
        tuple = tuple ++ &k.slots;
    }
    const reply_token_slots = [_]kinda.ResourceSlot{
        .{ .name = kinda.callback_runtime.ReplyToken.resource_name, .t = &kinda.callback_runtime.ReplyToken.resource.resource_type, .dtor = beam.destroy_do_nothing },
    };
    tuple = tuple ++ &reply_token_slots;
    break :blk tuple;
};

pub fn open_all(env: beam.env) void {
    inline for (allKinds) |k| {
        k.open_all(env);
    }
    inline for (kindAliases) |pair| {
        kinda.aliasKind(pair[0], pair[1]);
    }
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
