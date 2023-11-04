pub const c = @cImport({
    @cDefine("_NO_CRT_STDIO_INLINE", "1");
    @cInclude("mlir-c/Beaver/wrapper.h");
});

pub const DiagnosticHandlerDeleteUserData = ?*const fn (?*anyopaque) callconv(.C) void;
pub const ExternalPassConstruct = ?*const fn (?*anyopaque) callconv(.C) ?*anyopaque;
pub const ExternalPassRun = ?*const fn (c.MlirOperation, c.MlirExternalPass, ?*anyopaque) callconv(.C) void;
pub const GenericCallback = ?*const fn (c.MlirContext, ?*anyopaque) callconv(.C) c.MlirLogicalResult;
pub const UnmanagedDenseResourceElementsAttrGetDeleteCallback = ?*const fn (?*anyopaque, ?*const anyopaque, usize, usize) callconv(.C) void;
pub const AffineMapCompressUnusedSymbolsPopulateResult = ?*const fn (?*anyopaque, isize, c.MlirAffineMap) callconv(.C) void;
pub const SymbolTableWalkSymbolTablesCallback = ?*const fn (c.MlirOperation, bool, ?*anyopaque) callconv(.C) void;
pub usingnamespace c;
