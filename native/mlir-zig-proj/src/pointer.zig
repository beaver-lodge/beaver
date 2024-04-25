const beam = @import("beam");
const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const result = @import("result.zig");

fn get_null(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    return try mlir_capi.OpaquePtr.resource.make(env, null);
}

pub const nifs = .{result.nif("beaver_raw_get_null_ptr", 0, get_null).entry};
