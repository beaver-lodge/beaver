const std = @import("std");
const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const result = @import("kinda").result;
const StringRefCollector = @import("string_ref.zig").StringRefCollector;

fn get_all_registered_ops(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
    var col = StringRefCollector.init(env);
    c.beaverContextGetOps(ctx, StringRefCollector.append, @constCast(@ptrCast(@alignCast(&col))));
    return try col.collect_and_destroy();
}

fn get_registered_dialects(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
    var col = StringRefCollector.init(env);
    c.beaverContextGetDialects(ctx, StringRefCollector.append, @constCast(@ptrCast(@alignCast(&col))));
    return try col.collect_and_destroy();
}

pub const nifs = .{
    result.nif("beaver_raw_registered_ops", 1, get_all_registered_ops).entry,
    result.nif("beaver_raw_registered_dialects", 1, get_registered_dialects).entry,
};
