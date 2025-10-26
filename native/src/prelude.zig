pub const c = @cImport({
    @cDefine("_NO_CRT_STDIO_INLINE", "1");
    @cInclude("mlir-c/Beaver/wrapper.h");
});
const kinda = @import("kinda");
const e = kinda.erl_nif;
const nifPrefix = "Elixir.Beaver.MLIR.CAPI.";
pub const allKinds = @import("mlir_capi.zig").allKinds;
pub fn nif(comptime name: anytype) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(allKinds, c, name, .{ .nif_name = nifPrefix ++ name });
}
pub fn nifDirtyCPU(comptime name: []const u8, comptime nif_name: ?[]const u8) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(allKinds, c, name, .{ .flags = e.ERL_NIF_DIRTY_JOB_CPU_BOUND, .nif_name = nifPrefix ++ (if (nif_name) |v| v else name) });
}
pub fn nifDirtyIO(comptime name: []const u8, comptime nif_name: ?[]const u8) e.ErlNifFunc {
    @setEvalBranchQuota(10000);
    return kinda.NIFFunc(allKinds, c, name, .{ .flags = e.ERL_NIF_DIRTY_JOB_IO_BOUND, .nif_name = nifPrefix ++ (if (nif_name) |v| v else name) });
}

const result = @import("kinda").result;
pub fn beaverRawNIF(worker: anytype, comptime field_name: []const u8, comptime arity: usize) e.ErlNifFunc {
    return result.nif("beaver_raw_" ++ field_name, arity, @field(worker, field_name)).entry;
}
