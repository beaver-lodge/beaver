const std = @import("std");
const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const debug_print = @import("std").debug.print;
const result = @import("kinda").result;
const diagnostic = @import("diagnostic.zig");
const Token = @import("logical_mutex.zig").Token;

const BeaverPass = extern struct {
    handler: beam.pid,
    fn construct(_: ?*anyopaque) callconv(.C) void {}
    fn destruct(userData: ?*anyopaque) callconv(.C) void {
        const ptr: *@This() = @ptrCast(@alignCast(userData));
        beam.allocator.destroy(ptr);
    }
    fn initialize(_: mlir_capi.Context.T, _: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        return c.mlirLogicalResultSuccess();
    }
    fn clone(userData: ?*anyopaque) callconv(.C) ?*anyopaque {
        const old: *@This() = @ptrCast(@alignCast(userData));
        const new = beam.allocator.create(@This()) catch unreachable;
        new.* = old.*;
        return new;
    }
    const Error = error{ @"Fail to allocate BEAM environment", @"Fail to send message to pass server", @"Fail to run a pass implemented in Elixir" };
    fn do_run(op: mlir_capi.Operation.T, userData: ?*anyopaque) !void {
        const ud: *@This() = @ptrCast(@alignCast(userData));
        const env = e.enif_alloc_env() orelse return Error.@"Fail to allocate BEAM environment";
        defer e.enif_clear_env(env);
        const handler = ud.*.handler;
        var tuple_slice: []beam.term = try beam.allocator.alloc(beam.term, 3);
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_atom(env, "run");
        tuple_slice[1] = try mlir_capi.Operation.resource.make(env, op);
        var token = Token{};
        tuple_slice[2] = try beam.make_ptr_resource_wrapped(env, &token);
        if (!beam.send(env, handler, beam.make_tuple(env, tuple_slice))) {
            return Error.@"Fail to send message to pass server";
        }
        if (c.beaverLogicalResultIsFailure(token.wait_logical())) return Error.@"Fail to run a pass implemented in Elixir";
    }
    fn run(op: mlir_capi.Operation.T, pass: c.MlirExternalPass, userData: ?*anyopaque) callconv(.C) void {
        if (do_run(op, userData)) |_| {} else |err| {
            c.mlirEmitError(c.mlirOperationGetLocation(op), @errorName(err));
            c.mlirExternalPassSignalFailure(pass);
        }
    }
};

pub fn do_create(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const name = try mlir_capi.StringRef.resource.fetch(env, args[0]);
    const argument = try mlir_capi.StringRef.resource.fetch(env, args[1]);
    const description = try mlir_capi.StringRef.resource.fetch(env, args[2]);
    const op_name = try mlir_capi.StringRef.resource.fetch(env, args[3]);
    const handler: beam.pid = try beam.get_pid(env, args[4]);

    const typeIDAllocator = c.mlirTypeIDAllocatorCreate();
    defer c.mlirTypeIDAllocatorDestroy(typeIDAllocator);
    const passID = c.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    const nDependentDialects = 0;
    const dependentDialects = null;
    const bp: *BeaverPass = try beam.allocator.create(BeaverPass);
    bp.* = BeaverPass{ .handler = handler };
    // use this function to avoid ABI issue
    const ep = c.beaverPassCreate(
        BeaverPass.construct,
        BeaverPass.destruct,
        BeaverPass.initialize,
        BeaverPass.clone,
        BeaverPass.run,
        passID,
        name,
        argument,
        description,
        op_name,
        nDependentDialects,
        dependentDialects,
        bp,
    );
    return try mlir_capi.Pass.resource.make(env, ep);
}

pub const nifs = .{
    result.nif("beaver_raw_create_mlir_pass", 5, do_create).entry,
};
