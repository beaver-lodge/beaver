const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const diagnostic = @import("diagnostic.zig");
const c = prelude.c;

const policy = @embedFile("capi_policy.txt");
const declarations = @typeInfo(c).@"struct".decls;

fn policyContains(comptime category: []const u8, comptime name: []const u8) bool {
    var lines = std.mem.splitScalar(u8, policy, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        var fields = std.mem.splitScalar(u8, line, ' ');
        const line_category = fields.next() orelse continue;
        const line_name = fields.next() orelse continue;
        if (std.mem.eql(u8, category, line_category) and std.mem.eql(u8, name, line_name)) {
            return true;
        }
    }
    return false;
}

fn isCandidate(comptime name: []const u8) bool {
    const has_supported_prefix = std.mem.startsWith(u8, name, "mlir") or
        std.mem.startsWith(u8, name, "beaver");
    return has_supported_prefix and
        !policyContains("exclude", name) and
        !policyContains("callback_bridge", name);
}

fn isFunction(comptime name: []const u8) bool {
    return switch (@typeInfo(@TypeOf(@field(c, name)))) {
        .@"fn" => true,
        else => false,
    };
}

fn hasDiagnostics(comptime name: []const u8) bool {
    return policyContains("diagnostics", name) or std.mem.endsWith(u8, name, "GetChecked");
}

fn entryCount() usize {
    @setEvalBranchQuota(100_000_000);
    var count: usize = 0;
    for (declarations) |declaration| {
        if (!isCandidate(declaration.name)) continue;
        if (!isFunction(declaration.name)) continue;

        count += if (hasDiagnostics(declaration.name))
            2
        else if (policyContains("dirty_cpu_io", declaration.name))
            3
        else
            1;
    }
    return count;
}

pub const nifs = blk: {
    @setEvalBranchQuota(100_000_000);
    var entries: [entryCount()]e.ErlNifFunc = undefined;
    var index: usize = 0;

    for (declarations) |declaration| {
        const name = declaration.name;
        if (!isCandidate(name)) continue;
        if (!isFunction(name)) continue;

        entries[index] = prelude.nif(name);
        index += 1;

        if (hasDiagnostics(name)) {
            entries[index] = diagnostic.WithDiagnosticsNIF(name);
            index += 1;
        } else if (policyContains("dirty_cpu_io", name)) {
            entries[index] = prelude.nifDirtyCPU(name, name ++ "_dirty_cpu");
            entries[index + 1] = prelude.nifDirtyIO(name, name ++ "_dirty_io");
            index += 2;
        }
    }

    if (index != entries.len) @compileError("reflected C API registry size mismatch");
    break :blk entries;
};
