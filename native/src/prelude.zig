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

const kinda = @import("kinda");
const e = @import("erl_nif");
const nifPrefix = "Elixir.Beaver.MLIR.CAPI.";
pub fn N(comptime Kinds: anytype, c_: anytype, comptime name: anytype) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(Kinds, c_, name, .{ .nif_name = nifPrefix ++ name });
}
pub fn D_CPU(comptime Kinds: anytype, c_: anytype, comptime name: []const u8, comptime nif_name: ?[]const u8) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(Kinds, c_, name, .{ .flags = e.ERL_NIF_DIRTY_JOB_CPU_BOUND, .nif_name = nifPrefix ++ (if (nif_name) |v| v else name) });
}
pub fn D_IO(comptime Kinds: anytype, c_: anytype, comptime name: []const u8, comptime nif_name: ?[]const u8) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(Kinds, c_, name, .{ .flags = e.ERL_NIF_DIRTY_JOB_IO_BOUND, .nif_name = nifPrefix ++ (if (nif_name) |v| v else name) });
}
pub const K = @import("mlir_capi.zig").allKinds;
