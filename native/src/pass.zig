const std = @import("std");
const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
const c = @import("prelude.zig");
const e = @import("erl_nif");
const kinda = @import("kinda");
const result = @import("kinda").result;
const diagnostic = @import("diagnostic.zig");

const beaverPassCreateWrap = kinda.BangFunc(c.K, c, "beaverPassCreate").wrap_ret_call;
threadlocal var typeIDAllocator: ?mlir_capi.TypeIDAllocator.T = null;
const CallbackDispatcher = struct {
    handler: beam.pid,
    env: beam.env,
    id: beam.term,
    callbacks: struct { construct: ?beam.term = null, destruct: ?beam.term = null, initialize: ?beam.term = null, clone: ?beam.term = null, run: ?beam.term = null },
    fn construct(_: ?*anyopaque) callconv(.C) void {
        // support construct callback will require pass creation to be async, do nothing to keep it simple for now
    }
    fn destruct(userData: ?*anyopaque) callconv(.C) void {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb(this, "destruct", env, .{}) catch unreachable;
        if (c.beaverLogicalResultIsFailure(res)) @panic("Fail to destruct a pass implemented in Elixir");
        e.enif_free_env(this.env);
        beam.allocator.destroy(this);
    }
    fn initialize(ctx: mlir_capi.Context.T, userData: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb(this, "initialize", env, .{mlir_capi.Context.resource.make(env, ctx) catch unreachable}) catch return c.mlirLogicalResultFailure();
        if (c.beaverLogicalResultIsFailure(res)) {
            const loc = c.mlirLocationUnknownGet(ctx);
            c.mlirEmitError(loc, "Fail to initialize a pass implemented in Elixir");
        }
        return res;
    }
    const callback_names = .{ "destruct", "initialize", "clone", "run" };
    fn clone(userData: ?*anyopaque) callconv(.C) ?*anyopaque {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const new = @This().init(this.handler) catch unreachable;
        inline for (callback_names) |f| {
            const cb: ?beam.term = @field(this.*.callbacks, f);
            if (cb != null) {
                @field(new.*.callbacks, f) = e.enif_make_copy(new.*.env, cb.?);
            }
        }
        const env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb(new, "clone", env, .{e.enif_make_copy(new.*.env, this.id)}) catch unreachable;
        if (c.beaverLogicalResultIsFailure(res)) @panic("Fail to clone a pass implemented in Elixir");
        return new;
    }
    const Token = @import("logical_mutex.zig").Token;
    const Error = error{ @"Fail to allocate BEAM environment", @"Fail to send message to pass server", @"Fail to run a pass implemented in Elixir", @"External pass must be run on non-scheduler thread to prevent deadlock" };
    fn forward_cb(this: *@This(), comptime callback: []const u8, env: beam.env, args: anytype) !mlir_capi.LogicalResult.T {
        if (e.enif_thread_type() != e.ERL_NIF_THR_UNDEFINED) {
            return Error.@"External pass must be run on non-scheduler thread to prevent deadlock";
        }
        const cb = @field(this.*.callbacks, callback);
        if (cb == null) {
            e.enif_free_env(env);
            return c.mlirLogicalResultSuccess();
        } else {
            var token = Token{};
            var buffer = std.ArrayList(beam.term).init(beam.allocator);
            defer buffer.deinit();
            try buffer.append(beam.make_atom(env, callback));
            try buffer.append(try beam.make_ptr_resource_wrapped(env, &token));
            try buffer.append(e.enif_make_copy(env, cb.?));
            try buffer.append(e.enif_make_copy(env, this.id));
            inline for (args) |arg| {
                try buffer.append(arg);
            }
            const msg = beam.make_tuple(env, buffer.items);
            errdefer e.enif_free_env(env);
            if (!beam.send_advanced(env, this.*.handler, env, msg)) {
                return Error.@"Fail to send message to pass server";
            }
            return token.wait_logical();
        }
    }
    fn run(op: mlir_capi.Operation.T, pass: c.MlirExternalPass, userData: ?*anyopaque) callconv(.C) void {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const env = e.enif_alloc_env() orelse unreachable;
        const op_ = mlir_capi.Operation.resource.make(env, op) catch return c.mlirExternalPassSignalFailure(pass);
        if (forward_cb(this, "run", env, .{op_})) |res| {
            if (c.beaverLogicalResultIsFailure(res)) {
                c.mlirEmitError(c.mlirOperationGetLocation(op), "Fail to run a pass implemented in Elixir");
                c.mlirExternalPassSignalFailure(pass);
            }
        } else |err| {
            c.mlirEmitError(c.mlirOperationGetLocation(op), @errorName(err));
            c.mlirExternalPassSignalFailure(pass);
        }
    }
    fn init(handler: beam.pid) !*@This() {
        const this = try beam.allocator.create(@This());
        const this_env = e.enif_alloc_env() orelse return Error.@"Fail to allocate BEAM environment";
        this.* = @This(){ .handler = handler, .env = this_env, .id = e.enif_make_unique_integer(this_env, e.ERL_NIF_UNIQUE_POSITIVE), .callbacks = .{} };
        return this;
    }
    // use this function to avoid ABI issue
    fn create_mlir_pass(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const name = try mlir_capi.StringRef.resource.fetch(env, args[0]);
        const argument = try mlir_capi.StringRef.resource.fetch(env, args[1]);
        const description = try mlir_capi.StringRef.resource.fetch(env, args[2]);
        const op_name = try mlir_capi.StringRef.resource.fetch(env, args[3]);
        var handler: beam.pid = undefined;
        if (e.enif_self(env, &handler) == null) unreachable;
        if (typeIDAllocator == null) {
            typeIDAllocator = c.mlirTypeIDAllocatorCreate();
        }
        const passID = c.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator.?);
        const nDependentDialects = 0;
        const dependentDialects = null;
        const this: *@This() = try @This().init(handler);
        inline for (callback_names, 4..) |f, i| {
            const cb = args[i];
            if (!beam.is_nil2(env, cb)) {
                @field(this.*.callbacks, f) = e.enif_make_copy(this.env, args[i]);
            }
        }
        return beaverPassCreateWrap(env, .{ construct, destruct, initialize, clone, run, passID, name, argument, description, op_name, nDependentDialects, dependentDialects, this });
    }
};

const mlirPassManagerRunOnOpWrap = kinda.BangFunc(c.K, c, "mlirPassManagerRunOnOp").wrap_ret_call;
const ManagerRunner = struct {
    const Error = error{ @"fail to allocate BEAM environment", @"fail to send message to pm caller", @"fail get caller's self pid" };
    pid: beam.pid,
    pm: mlir_capi.PassManager.T,
    op: ?mlir_capi.Operation.T = null,
    fn init(env: beam.env) !*@This() {
        const this = try beam.allocator.create(@This());
        if (e.enif_self(env, &this.*.pid) == null) {
            return Error.@"fail get caller's self pid";
        }
        return this;
    }
    fn run_with_diagnostics(this: @This()) !void {
        const env = e.enif_alloc_env() orelse return Error.@"fail to allocate BEAM environment";
        const ctx = c.mlirOperationGetContext(this.op.?);
        const args = .{ this.pm, this.op.? };
        errdefer e.enif_free_env(env);
        if (!beam.send_advanced(env, this.pid, env, try diagnostic.call_with_diagnostics(env, ctx, mlirPassManagerRunOnOpWrap, .{ env, args }))) {
            return Error.@"fail to send message to pm caller";
        }
    }
    fn run_and_send(worker: ?*anyopaque) callconv(.C) void {
        const this: ?*@This() = @ptrCast(@alignCast(worker));
        defer beam.allocator.destroy(this.?);
        if (run_with_diagnostics(this.?.*)) |_| {} else |err| {
            c.mlirEmitError(c.mlirOperationGetLocation(this.?.op.?), @errorName(err));
        }
    }
    fn run_pm_on_op(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const this = try init(env);
        errdefer beam.allocator.destroy(this);
        this.*.pm = try mlir_capi.PassManager.resource.fetch(env, args[0]);
        this.*.op = try mlir_capi.Operation.resource.fetch(env, args[1]);
        const ctx = c.mlirOperationGetContext(this.op.?);
        if (c.beaverContextAddWork(ctx, @This().run_and_send, @ptrCast(@constCast(this)))) {
            return beam.make_atom(env, "async");
        } else {
            defer beam.allocator.destroy(this);
            return try diagnostic.call_with_diagnostics(env, ctx, mlirPassManagerRunOnOpWrap, .{ env, .{ this.pm, this.op.? } });
        }
    }
    fn send_ok(this: *@This()) !void {
        const env = e.enif_alloc_env() orelse return Error.@"fail to allocate BEAM environment";
        if (!beam.send_advanced(env, this.pid, env, beam.make_ok(env))) {
            return Error.@"fail to send message to pm caller";
        }
    }
    fn destroyManager(worker: ?*anyopaque) callconv(.C) void {
        const this: ?*@This() = @ptrCast(@alignCast(worker));
        defer beam.allocator.destroy(this.?);
        c.mlirPassManagerDestroy(this.?.*.pm);
        send_ok(this.?) catch unreachable;
    }
    fn destroy(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const this = try init(env);
        errdefer beam.allocator.destroy(this);
        this.*.pm = try mlir_capi.PassManager.resource.fetch(env, args[0]);
        const ctx = c.beaverPassManagerGetContext(this.pm);
        if (c.beaverContextAddWork(ctx, destroyManager, @ptrCast(@constCast(this)))) {
            return beam.make_atom(env, "async");
        } else {
            defer beam.allocator.destroy(this);
            c.mlirPassManagerDestroy(this.pm);
            return beam.make_ok(env);
        }
    }
};

var mutex: std.Thread.Mutex = .{};
var all_passes_registered = false;
pub fn register_all_passes() void {
    mutex.lock();
    defer mutex.unlock();
    if (!all_passes_registered) {
        all_passes_registered = true;
        c.mlirRegisterAllPasses();
    }
}

pub const nifs = .{
    result.nif("beaver_raw_create_mlir_pass", 8, CallbackDispatcher.create_mlir_pass).entry,
    result.nif("beaver_raw_run_pm_on_op_async", 2, ManagerRunner.run_pm_on_op).entry,
    result.nif("beaver_raw_destroy_pm_async", 1, ManagerRunner.destroy).entry,
};
