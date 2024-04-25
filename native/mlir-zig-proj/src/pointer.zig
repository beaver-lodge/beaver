const beam = @import("beam");
const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const result = @import("result.zig");
const kinda = @import("kinda");

pub const PtrOwner = extern struct {
    pub const Kind = kinda.ResourceKind(@This(), "Elixir.Beaver.Native.PtrOwner");
    ptr: mlir_capi.OpaquePtr.T,
    extern fn free(ptr: ?*anyopaque) void;
    pub fn destroy(_: beam.env, resource_ptr: ?*anyopaque) callconv(.C) void {
        const this_ptr: *@This() = @ptrCast(@alignCast(resource_ptr));
        @import("std").debug.print("destroy {}.\n", .{this_ptr});
        free(this_ptr.*.ptr);
    }
};

fn get_null(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    return try mlir_capi.OpaquePtr.resource.make(env, null);
}

fn own_opaque_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    var ptr = try mlir_capi.OpaquePtr.resource.fetch(env, args[0]);
    var owner: PtrOwner = .{ .ptr = ptr };
    return try PtrOwner.Kind.resource.make(env, owner);
}

fn read_opaque_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    var ptr = try mlir_capi.OpaquePtr.resource.fetch(env, args[0]);
    var len = try mlir_capi.USize.resource.fetch(env, args[1]);
    if (ptr == null) {
        return beam.make_error_binary(env, "ptr is null");
    }
    const slice = @as(mlir_capi.U8.Array.T, @ptrCast(ptr))[0..len];
    return beam.make_slice(env, slice);
}

pub const nifs = .{ result.nif("beaver_raw_get_null_ptr", 0, get_null).entry, result.nif("beaver_raw_own_opaque_ptr", 1, own_opaque_ptr).entry, result.nif("beaver_raw_read_opaque_ptr", 2, read_opaque_ptr).entry };
