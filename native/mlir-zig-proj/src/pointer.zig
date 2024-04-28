const beam = @import("beam");
const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const kinda = @import("kinda");
const result = kinda.result;

extern fn free(ptr: mlir_capi.OpaquePtr.T) void;

fn get_null(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    return try mlir_capi.OpaquePtr.resource.make(env, null);
}

fn deallocate(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const Error = error{NullPointer};
    var ptr: mlir_capi.OpaquePtr.T = try mlir_capi.OpaquePtr.resource.fetch(env, args[0]);
    if (ptr) |p| {
        free(p);
        return beam.make_ok(env);
    } else {
        return Error.NullPointer;
    }
}

fn read_opaque_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    var ptr = try mlir_capi.OpaquePtr.resource.fetch(env, args[0]);
    var len = try mlir_capi.USize.resource.fetch(env, args[1]);
    const Error = error{NullPointer};
    if (ptr == null) {
        return Error.NullPointer;
    }
    const slice = @as(mlir_capi.U8.Array.T, @ptrCast(ptr))[0..len];
    return beam.make_slice(env, slice);
}

pub const nifs = .{ result.nif("beaver_raw_get_null_ptr", 0, get_null).entry, result.nif("beaver_raw_deallocate_opaque_ptr", 1, deallocate).entry, result.nif("beaver_raw_read_opaque_ptr", 2, read_opaque_ptr).entry };
