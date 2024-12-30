const std = @import("std");
const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const debug_print = @import("std").debug.print;
const kinda = @import("kinda");
const result = @import("kinda").result;
const diagnostic = @import("diagnostic.zig");
const DiagnosticAggregator = diagnostic.DiagnosticAggregator;
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
        if (!beam.send_advanced(env, handler, env, beam.make_tuple(env, tuple_slice))) {
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

const WorkerError = error{ @"fail to allocate BEAM environment", @"fail to send message to pm caller", @"fail get caller's self pid" };

// we only use the return functionality of BangFunc here because we are not fetching resources here
const mlirPassManagerRunOnOpWrap = kinda.BangFunc(c.K, c, "mlirPassManagerRunOnOp").wrap_ret_call;
const PassManagerRunner = extern struct {
    pid: beam.pid,
    pm: mlir_capi.PassManager.T,
    op: mlir_capi.Operation.T,
    fn run_with_diagnostics(this: @This()) !void {
        const env = e.enif_alloc_env() orelse return WorkerError.@"fail to allocate BEAM environment";
        const ctx = c.mlirOperationGetContext(this.op);
        const args = .{ this.pm, this.op };
        if (!beam.send_advanced(env, this.pid, env, try diagnostic.call_with_diagnostics(env, ctx, mlirPassManagerRunOnOpWrap, .{ env, args }))) {
            return WorkerError.@"fail to send message to pm caller";
        }
    }
    fn run_and_send(worker: ?*anyopaque) callconv(.C) void {
        const this: ?*@This() = @ptrCast(@alignCast(worker));
        defer beam.allocator.destroy(this.?);
        if (run_with_diagnostics(this.?.*)) |_| {} else |err| {
            c.mlirEmitError(c.mlirOperationGetLocation(this.?.*.op), @errorName(err));
        }
    }
};

pub fn run_pm_on_op(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const w = try beam.allocator.create(PassManagerRunner);
    if (e.enif_self(env, &w.*.pid) == null) {
        return WorkerError.@"fail get caller's self pid";
    }
    w.*.pm = try mlir_capi.PassManager.resource.fetch(env, args[0]);
    w.*.op = try mlir_capi.Operation.resource.fetch(env, args[1]);
    const ctx = c.mlirOperationGetContext(w.op);
    if (c.beaverContextAddWork(ctx, PassManagerRunner.run_and_send, @ptrCast(@constCast(w)))) {
        return beam.make_ok(env);
    } else {
        defer beam.allocator.destroy(w);
        return try diagnostic.call_with_diagnostics(env, ctx, mlirPassManagerRunOnOpWrap, .{ env, .{ w.*.pm, w.*.op } });
    }
}

pub const nifs = .{
    result.nif("beaver_raw_create_mlir_pass", 5, do_create).entry,
    result.nif("beaver_raw_run_pm_on_op_async", 2, run_pm_on_op).entry,
};
