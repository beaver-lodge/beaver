pub const c = @cImport({
    @cDefine("_NO_CRT_STDIO_INLINE", "1");
    @cInclude("mlir-c/Beaver/wrapper.h");
});

pub const MlirDiagnosticHandlerDeleteUserData = ?*const fn (?*anyopaque) callconv(.C) void;
pub const MlirExternalPassConstruct = ?*const fn (?*anyopaque) callconv(.C) ?*anyopaque;
pub const MlirExternalPassRun = ?*const fn (c.MlirOperation, c.MlirExternalPass, ?*anyopaque) callconv(.C) void;
pub const MlirGenericCallback = ?*const fn (c.MlirContext, ?*anyopaque) callconv(.C) c.MlirLogicalResult;
pub const MlirUnmanagedDenseResourceElementsAttrGetDeleteCallback = ?*const fn (?*anyopaque, ?*const anyopaque, usize, usize) callconv(.C) void;
pub const MlirAffineMapCompressUnusedSymbolsPopulateResult = ?*const fn (?*anyopaque, isize, c.MlirAffineMap) callconv(.C) void;
pub const MlirSymbolTableWalkSymbolTablesCallback = ?*const fn (c.MlirOperation, bool, ?*anyopaque) callconv(.C) void;
pub usingnamespace c;
