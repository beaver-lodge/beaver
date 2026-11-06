const std = @import("std");
const e = @import("kinda").erl_nif;

// Only ordinary C functions can be exposed as JIT symbols. The PID comparison
// macro is the one generic adapter supported by our signature decoder.
pub const functions = blk: {
    @setEvalBranchQuota(100000);
    const declarations = @typeInfo(e).@"struct".decls;
    var names: [declarations.len][]const u8 = undefined;
    var count: usize = 0;
    for (declarations) |decl| {
        const name = decl.name;
        if (!std.mem.startsWith(u8, name, "enif_")) continue;
        // These macros cannot be translated, or aren't part of the JIT API.
        if (std.mem.eql(u8, name, "enif_make_pid") or std.mem.eql(u8, name, "enif_select_x") or
            std.mem.eql(u8, name, "enif_dlopen") or std.mem.eql(u8, name, "enif_dlsym")) continue;
        const info = @typeInfo(@TypeOf(@field(e, name)));
        if (info != .@"fn") continue;
        if (info.@"fn".is_generic and !std.mem.eql(u8, name, "enif_compare_pids")) continue;
        names[count] = name;
        count += 1;
    }
    std.mem.sort([]const u8, names[0..count], {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
    break :blk names[0..count].*;
};
