const std = @import("std");
const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const debug_print = @import("std").debug.print;
const result = @import("result.zig");

fn get_all_registered_ops(env: beam.env) !beam.term {
    const ctx = get_context_load_all_dialects();
    defer c.mlirContextDestroy(ctx);
    const num_op: isize = c.beaverGetNumRegisteredOperations(ctx);
    var i: isize = 0;
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(num_op));
    defer beam.allocator.free(ret);
    while (i < num_op) : ({
        i += 1;
    }) {
        const registered_op_name = c.beaverGetRegisteredOperationName(ctx, i);
        const dialect_name = c.beaverRegisteredOperationNameGetDialectName(registered_op_name);
        const op_name = c.beaverRegisteredOperationNameGetOpName(registered_op_name);
        var tuple_slice: []beam.term = try beam.allocator.alloc(beam.term, 2);
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_slice(env, dialect_name.data[0..dialect_name.length]);
        tuple_slice[1] = beam.make_slice(env, op_name.data[0..op_name.length]);
        ret[@intCast(i)] = beam.make_tuple(env, tuple_slice);
    }
    return beam.make_term_list(env, ret);
}

fn get_context_load_all_dialects() mlir_capi.Context.T {
    const ctx = c.mlirContextCreate();
    var registry = c.mlirDialectRegistryCreate();
    c.mlirRegisterAllDialects(registry);
    c.mlirContextAppendDialectRegistry(ctx, registry);
    c.mlirDialectRegistryDestroy(registry);
    c.mlirContextLoadAllAvailableDialects(ctx);
    return ctx;
}

fn get_all_registered_ops2(env: beam.env, ctx: mlir_capi.Context.T, dialect: mlir_capi.StringRef.T) !beam.term {
    var num_op: usize = 0;
    // TODO: refactor this dirty trick
    var names: [300]c.MlirRegisteredOperationName = undefined;
    c.beaverRegisteredOperationsOfDialect(ctx, dialect, &names, &num_op);
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(num_op));
    defer beam.allocator.free(ret);
    var i: usize = 0;
    while (i < num_op) : ({
        i += 1;
    }) {
        const registered_op_name = names[i];
        const op_name = c.beaverRegisteredOperationNameGetOpName(registered_op_name);
        ret[@intCast(i)] = beam.make_c_string_charlist(env, op_name.data);
    }
    return beam.make_term_list(env, ret);
}

pub export fn beaver_raw_registered_ops(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    return get_all_registered_ops(env) catch beam.make_error_binary(env, "launching nif");
}

fn get_registered_dialects(env: beam.env) !beam.term {
    const ctx = get_context_load_all_dialects();
    defer c.mlirContextDestroy(ctx);
    var num_dialects: usize = 0;
    // TODO: refactor this dirty trick
    var names: [300]mlir_capi.StringRef.T = undefined;
    c.beaverRegisteredDialects(ctx, &names, &num_dialects);
    if (num_dialects == 0) {
        return beam.make_error_binary(env, "no dialects found");
    }
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(num_dialects));
    defer beam.allocator.free(ret);
    var i: usize = 0;
    while (i < num_dialects) : ({
        i += 1;
    }) {
        ret[@intCast(i)] = beam.make_c_string_charlist(env, names[i].data);
    }
    return beam.make_term_list(env, ret);
}

pub export fn beaver_raw_registered_ops_of_dialect(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var dialect: mlir_capi.StringRef.T = undefined;
    var ctx: mlir_capi.Context.T = undefined;
    if (beam.fetch_resource(mlir_capi.Context.T, env, mlir_capi.Context.resource.t, args[0])) |value| {
        ctx = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for context, expected: mlir_capi.Context.T");
    }
    if (beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[1])) |value| {
        dialect = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for dialect, expected: mlir_capi.StringRef.T");
    }
    return get_all_registered_ops2(env, ctx, dialect) catch beam.make_error_binary(env, "launching nif");
}

pub export fn beaver_raw_registered_dialects(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    return get_registered_dialects(env) catch beam.make_error_binary(env, "launching nif");
}

pub export fn beaver_raw_get_context_load_all_dialects(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    var ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.Context.resource.t, @sizeOf(mlir_capi.Context.T));
    const RType = mlir_capi.Context.T;
    var obj: *RType = undefined;
    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(@alignCast(ptr));
        obj.* = get_context_load_all_dialects();
    }
    return e.enif_make_resource(env, ptr);
}
