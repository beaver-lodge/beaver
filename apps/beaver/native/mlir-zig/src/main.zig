const std = @import("std");
const testing = std.testing;

export fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try testing.expect(add(3, 7) == 10);
}
const beam = @import("beam.zig");
const e = @import("erl_nif.zig");
const fizz = @import("mlir.fizz.gen.zig");
pub const c = fizz.c;

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
    fizz.open_generated_resource_types(env);
    return 0;
}

fn get_all_registered_ops(env: beam.env) !beam.term {
    const ctx = c.mlirContextCreate();
    c.mlirRegisterAllDialects(ctx);
    const num_op: isize = c.beaverGetNumRegisteredOperations(ctx);
    var i: isize = 0;
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(usize, num_op));
    while (i < num_op) : ({
        i += 1;
    }) {
        const registered_op_name = c.beaverGetRegisteredOperationName(ctx, i);
        const dialect_name = c.beaverRegisteredOperationNameGetDialectName(registered_op_name);
        const op_name = c.beaverRegisteredOperationNameGetOpName(registered_op_name);
        var tuple_slice: []beam.term = try beam.allocator.alloc(beam.term, 2);
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_cstring_charlist(env, dialect_name.data);
        tuple_slice[1] = beam.make_cstring_charlist(env, op_name.data);
        ret[@intCast(usize, i)] = beam.make_tuple(env, tuple_slice);
    }
    c.mlirContextDestroy(ctx);
    return beam.make_term_list(env, ret);
}
export fn registered_ops(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    return get_all_registered_ops(env) catch beam.make_error_binary(env, "launching nif");
}

export fn resource_cstring_to_term_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    const T = [*c]const u8;
    var arg0: T = undefined;
    if (beam.fetch_resource(T, env, fizz.resource_type__c_ptr_const_u8, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource");
    }
    return beam.make_cstring_charlist(env, arg0);
}

export fn resource_bool_to_term(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: bool = undefined;
    if (beam.fetch_resource(bool, env, fizz.resource_type_bool, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource");
    }
    return beam.make_bool(env, arg0);
}

export fn get_resource_bool(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var ptr: ?*anyopaque = e.enif_alloc_resource(fizz.resource_type_bool, @sizeOf(bool));
    var obj: *bool = undefined;
    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(*bool, @alignCast(@alignOf(*bool), ptr));
    }
    if (beam.get(bool, env, args[0])) |value| {
        obj.* = value;
        return e.enif_make_resource(env, ptr);
    } else |_| {
        return beam.make_error_binary(env, "launching nif");
    }
}

// create a C string resource by copying given binary
const mem = @import("std").mem;
// memory layout {address, real_binary, null}
export fn get_resource_c_string(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    const RType = [*c]u8;
    var bin: beam.binary = undefined;
    if (0 == e.enif_inspect_binary(env, args[0], &bin)) {
        return beam.make_error_binary(env, "not a binary");
    }
    var ptr: ?*anyopaque = e.enif_alloc_resource(fizz.resource_type__c_ptr_const_u8, @alignOf(RType) + bin.size + 1);
    var obj: *RType = undefined;
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
    var real_binary: RType = undefined;
    if (ptr == null) {
        unreachable();
    } else {
        real_binary = @ptrCast(RType, ptr);
        real_binary += @alignOf(RType);
        real_binary[bin.size] = 0;
        obj.* = real_binary;
    }
    mem.copy(u8, real_binary[0..bin.size], bin.data[0..bin.size]);
    return e.enif_make_resource(env, ptr);
}

pub export const handwritten_nifs = ([_]e.ErlNifFunc{
    e.ErlNifFunc{ .name = "registered_ops", .arity = 0, .fptr = registered_ops, .flags = 0 },
    e.ErlNifFunc{ .name = "resource_cstring_to_term_charlist", .arity = 1, .fptr = resource_cstring_to_term_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "resource_bool_to_term", .arity = 1, .fptr = resource_bool_to_term, .flags = 0 },
    e.ErlNifFunc{ .name = "get_resource_bool", .arity = 1, .fptr = get_resource_bool, .flags = 0 },
    e.ErlNifFunc{ .name = "get_resource_c_string", .arity = 1, .fptr = get_resource_c_string, .flags = 0 },
});

pub export const num_nifs = fizz.generated_nifs.len + handwritten_nifs.len;
pub export var nifs: [num_nifs]e.ErlNifFunc = undefined;

const entry = e.ErlNifEntry{
    .major = 2,
    .minor = 16,
    .name = "Elixir.Beaver.MLIR.CAPI",
    .num_of_funcs = num_nifs,
    .funcs = &(nifs[0]),
    .load = nif_load,
    .reload = null, // currently unsupported
    .upgrade = null, // currently unsupported
    .unload = null, // currently unsupported
    .vm_variant = "beam.vanilla",
    .options = 1,
    .sizeof_ErlNifResourceTypeInit = @sizeOf(e.ErlNifResourceTypeInit),
    .min_erts = "erts-13.0",
};

export fn nif_init() *const e.ErlNifEntry {
    var i: usize = 0;
    while (i < fizz.generated_nifs.len) : ({
        i += 1;
    }) {
        nifs[i] = fizz.generated_nifs[i];
    }
    var j: usize = 0;
    while (j < handwritten_nifs.len) : ({
        j += 1;
    }) {
        nifs[fizz.generated_nifs.len + j] = handwritten_nifs[j];
    }
    return &entry;
}
