pub const c = @cImport({
    @cDefine("_NO_CRT_STDIO_INLINE", "1");
    @cInclude("mlir-c/Beaver/wrapper.h");
});
const kinda = @import("kinda");
const e = kinda.erl_nif;
const nifPrefix = "Elixir.Beaver.MLIR.CAPI.";
pub fn nif(comptime Kinds: anytype, c_: anytype, comptime name: anytype) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(Kinds, c_, name, .{ .nif_name = nifPrefix ++ name });
}
pub fn nifDirtyCPU(comptime Kinds: anytype, c_: anytype, comptime name: []const u8, comptime nif_name: ?[]const u8) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(Kinds, c_, name, .{ .flags = e.ERL_NIF_DIRTY_JOB_CPU_BOUND, .nif_name = nifPrefix ++ (if (nif_name) |v| v else name) });
}
pub fn nifDirtyIO(comptime Kinds: anytype, c_: anytype, comptime name: []const u8, comptime nif_name: ?[]const u8) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(Kinds, c_, name, .{ .flags = e.ERL_NIF_DIRTY_JOB_IO_BOUND, .nif_name = nifPrefix ++ (if (nif_name) |v| v else name) });
}
pub const allKinds = @import("mlir_capi.zig").allKinds;
