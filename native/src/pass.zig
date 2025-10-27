const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const debug_print = @import("std").debug.print;
const result = @import("kinda").result;
const diagnostic = @import("diagnostic.zig");
const callback_names = .{ "construct", "destruct", "initialize", "clone", "run" };

threadlocal var typeIDAllocator: ?mlir_capi.TypeIDAllocator.T = null;
const CallbackDispatcher = struct {
    handler: beam.pid,
    env: beam.env,
    id: beam.term,
    callbacks: struct { construct: ?beam.term = null, destruct: ?beam.term = null, initialize: ?beam.term = null, clone: ?beam.term = null, run: ?beam.term = null },
    fn construct(userData: ?*anyopaque) callconv(.c) void {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb_and_consume_env(this, "construct", temp_env, .{}) catch unreachable;
        if (c.beaverLogicalResultIsFailure(res)) @panic("Fail to construct a pass implemented in Elixir");
    }
    fn destruct(userData: ?*anyopaque) callconv(.c) void {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb_and_consume_env(this, "destruct", temp_env, .{}) catch unreachable;
        if (c.beaverLogicalResultIsFailure(res)) @panic("Fail to destruct a pass implemented in Elixir");
        e.enif_free_env(this.env);
        beam.allocator.destroy(this);
    }
    fn initialize(ctx: mlir_capi.Context.T, userData: ?*anyopaque) callconv(.c) mlir_capi.LogicalResult.T {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb_and_consume_env(this, "initialize", temp_env, .{mlir_capi.Context.resource.make_kind(temp_env, ctx) catch unreachable}) catch return c.mlirLogicalResultFailure();
        if (c.beaverLogicalResultIsFailure(res)) {
            const loc = c.mlirLocationUnknownGet(ctx);
            c.mlirEmitError(loc, "Fail to initialize a pass implemented in Elixir");
        }
        return res;
    }
    fn clone(userData: ?*anyopaque) callconv(.c) ?*anyopaque {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const new_dispatcher = @This().init(this.handler) catch unreachable;
        inline for (callback_names) |f| {
            const cb: ?beam.term = @field(this.*.callbacks, f);
            if (cb != null) {
                @field(new_dispatcher.*.callbacks, f) = e.enif_make_copy(new_dispatcher.*.env, cb.?);
            }
        }
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb_and_consume_env(new_dispatcher, "clone", temp_env, .{e.enif_make_copy(new_dispatcher.*.env, this.id)}) catch unreachable;
        if (c.beaverLogicalResultIsFailure(res)) @panic("Fail to clone a pass implemented in Elixir");
        return new_dispatcher;
    }
    const Token = @import("logical_mutex.zig").Token;
    const Error = error{ @"Fail to allocate BEAM environment", @"Fail to send message to pass server", @"Fail to run a pass implemented in Elixir" };
    fn forward_cb_and_consume_env(this: *@This(), comptime callback: []const u8, temp_env: beam.env, args: anytype) !mlir_capi.LogicalResult.T {
        if (e.enif_thread_type() != e.ERL_NIF_THR_UNDEFINED) {
            @panic("External pass must be run on non-scheduler thread to prevent deadlock. Callback: " ++ callback);
        }
        const cb = @field(this.*.callbacks, callback);
        if (cb == null) {
            e.enif_free_env(temp_env);
            return c.mlirLogicalResultSuccess();
        } else {
            var token = Token{};
            var buffer = std.array_list.Managed(beam.term).init(beam.allocator);
            defer buffer.deinit();
            try buffer.append(beam.make_atom(temp_env, callback));
            try buffer.append(try beam.make_ptr_resource_wrapped(temp_env, &token));
            try buffer.append(e.enif_make_copy(temp_env, cb.?));
            try buffer.append(e.enif_make_copy(temp_env, this.id));
            inline for (args) |arg| {
                try buffer.append(arg);
            }
            const msg = beam.make_tuple(temp_env, buffer.items);
            errdefer e.enif_free_env(temp_env);
            if (!beam.send_advanced(temp_env, this.*.handler, temp_env, msg)) {
                return Error.@"Fail to send message to pass server";
            }
            const ret = token.wait_logical();
            // Transfer owner ship to last caller
            this.handler = ret.caller;
            return ret.result;
        }
    }
    fn run(op: mlir_capi.Operation.T, pass: c.MlirExternalPass, userData: ?*anyopaque) callconv(.c) void {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const op_kind = mlir_capi.Operation.resource.make_kind(temp_env, op) catch return c.mlirExternalPassSignalFailure(pass);
        if (forward_cb_and_consume_env(this, "run", temp_env, .{op_kind})) |res| {
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
};

const mlirPassManagerRunOnOpWrap = kinda.BangFunc(prelude.allKinds, c, "mlirPassManagerRunOnOp").wrap_ret_call;
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
    fn run_and_send(worker: ?*anyopaque) callconv(.c) void {
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

    fn destroyManager(worker: ?*anyopaque) callconv(.c) void {
        const this: ?*@This() = @ptrCast(@alignCast(worker));
        defer beam.allocator.destroy(this.?);
        c.mlirPassManagerDestroy(this.?.*.pm);
        const msg = "pm_destroy_done";
        const temp_env = e.enif_alloc_env() orelse @panic("fail to allocate BEAM environment: " ++ msg);
        if (!beam.send_advanced(temp_env, this.?.pid, temp_env, beam.make_atom(temp_env, msg))) {
            @panic("fail to send message to pm caller: " ++ msg);
        }
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

const beaverPassCreateWrap = kinda.BangFunc(prelude.allKinds, c, "beaverPassCreate").wrap_ret_call;
const PassCreator = struct {
    pid: beam.pid,
    name: mlir_capi.StringRef.T,
    argument: mlir_capi.StringRef.T,
    description: mlir_capi.StringRef.T,
    op_name: mlir_capi.StringRef.T,
    cb_map: std.StringHashMap(beam.term),

    fn init(env: beam.env, name: beam.term, argument: beam.term, description: beam.term, op_name: beam.term, callbacks: beam.term) !*@This() {
        const this = try beam.allocator.create(@This());
        errdefer beam.allocator.destroy(this);
        if (e.enif_self(env, &this.pid) == null) return error.FailToGetSelfPid;
        this.cb_map = std.StringHashMap(beam.term).init(beam.allocator);
        errdefer this.cb_map.deinit();
        this.* = .{
            .pid = this.pid,
            .name = try mlir_capi.StringRef.resource.fetch(env, name),
            .argument = try mlir_capi.StringRef.resource.fetch(env, argument),
            .description = try mlir_capi.StringRef.resource.fetch(env, description),
            .op_name = try mlir_capi.StringRef.resource.fetch(env, op_name),
            .cb_map = this.cb_map,
        };
        inline for (callback_names) |f| {
            const key = beam.make_atom(env, f);
            var cb: beam.term = undefined;
            if (e.enif_get_map_value(env, callbacks, key, &cb) <= 0) {
                @panic("fail to get cb from map:" ++ f);
            }
            if (!beam.is_nil2(env, cb)) {
                try this.cb_map.put(f, cb);
            }
        }
        return this;
    }
};

fn create_mlir_pass_sync(env: beam.env, creator: *const PassCreator) !beam.term {
    if (typeIDAllocator == null) {
        typeIDAllocator = c.mlirTypeIDAllocatorCreate();
    }
    const passID = c.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator.?);
    const nDependentDialects = 0;
    const dependentDialects = null;
    const dispatcher: *CallbackDispatcher = try CallbackDispatcher.init(creator.pid);
    inline for (callback_names) |f| {
        if (creator.cb_map.get(f)) |cb| {
            @field(dispatcher.callbacks, f) = e.enif_make_copy(dispatcher.env, cb);
        }
    }
    return beaverPassCreateWrap(env, .{
        CallbackDispatcher.construct,
        CallbackDispatcher.destruct,
        CallbackDispatcher.initialize,
        CallbackDispatcher.clone,
        CallbackDispatcher.run,
        passID,
        creator.name,
        creator.argument,
        creator.description,
        creator.op_name,
        nDependentDialects,
        dependentDialects,
        dispatcher,
    });
}

fn create_and_send_pass(worker: ?*anyopaque) callconv(.c) void {
    const creator: *PassCreator = @ptrCast(@alignCast(worker));
    defer beam.allocator.destroy(creator);
    const temp_env = e.enif_alloc_env() orelse unreachable;
    defer e.enif_free_env(temp_env);
    const pass = create_mlir_pass_sync(temp_env, creator) catch @panic("fail to create pass");
    if (!beam.send_advanced(null, creator.pid, temp_env, pass)) {
        @panic("fail to send pass");
    }
}

fn create_mlir_pass(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
    const creator = try PassCreator.init(env, args[1], args[2], args[3], args[4], args[5]);
    if (c.beaverContextAddWork(ctx, create_and_send_pass, @ptrCast(creator))) {
        return beam.make_atom(env, "async");
    } else {
        defer beam.allocator.destroy(creator);
        return create_mlir_pass_sync(env, creator);
    }
}

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
    result.nif("beaver_raw_create_mlir_pass", 6, create_mlir_pass).entry,
    result.nif("beaver_raw_run_pm_on_op_async", 2, ManagerRunner.run_pm_on_op).entry,
    result.nif("beaver_raw_destroy_pm_async", 1, ManagerRunner.destroy).entry,
};
