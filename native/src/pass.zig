const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const diagnostic = @import("diagnostic.zig");
const callback_names = .{ "construct", "destruct", "initialize", "clone", "run" };
const string_ref = @import("string_ref.zig");

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
        const env = e.enif_alloc_env() orelse unreachable;
        const new_dispatcher = @This().init(env, this.handler) catch unreachable;
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
            if (!beam.send_advanced(null, this.*.handler, temp_env, msg)) {
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
    fn init(env: beam.env, handler: beam.pid) !*@This() {
        const this = try beam.allocator.create(@This());
        this.* = @This(){ .handler = handler, .env = env, .id = e.enif_make_unique_integer(env, e.ERL_NIF_UNIQUE_POSITIVE), .callbacks = .{} };
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
    pub fn run_pm_on_op_async(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
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
        if (!beam.send_advanced(null, this.?.pid, temp_env, beam.make_atom(temp_env, msg))) {
            @panic("fail to send message to pm caller: " ++ msg);
        }
    }
    pub fn destroy_pm_async(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
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

const PassCreator = struct {
    pid: beam.pid,
    name: beam.term,
    argument: beam.term,
    description: beam.term,
    op_name: beam.term,
    cb_map: std.StringHashMap(beam.term),
    dispatcher_env: beam.env,

    fn init(pid: beam.pid, args: *const [4]beam.term) !*@This() {
        const this = try beam.allocator.create(@This());
        const env = e.enif_alloc_env() orelse unreachable;
        this.* = .{
            .dispatcher_env = env,
            .pid = pid,
            .name = e.enif_make_copy(env, args[0]),
            .argument = e.enif_make_copy(env, args[1]),
            .description = e.enif_make_copy(env, args[2]),
            .op_name = e.enif_make_copy(env, args[3]),
            .cb_map = std.StringHashMap(beam.term).init(beam.allocator),
        };
        errdefer this.deinit();
        return this;
    }
    fn deinit(self: *@This()) void {
        self.cb_map.deinit();
        beam.allocator.destroy(self);
    }
    const StructOfCallbacks = c.MlirExternalPassCallbacks{ .construct = CallbackDispatcher.construct, .destruct = CallbackDispatcher.destruct, .initialize = CallbackDispatcher.initialize, .clone = CallbackDispatcher.clone, .run = CallbackDispatcher.run };
    fn create_and_send_pass(worker: ?*anyopaque) callconv(.c) void {
        if (e.enif_thread_type() != e.ERL_NIF_THR_UNDEFINED) {
            @panic("Must apply pattern on non-scheduler thread to prevent deadlock");
        }
        const this: *@This() = @ptrCast(@alignCast(worker));
        defer this.deinit();
        if (typeIDAllocator == null) {
            typeIDAllocator = c.mlirTypeIDAllocatorCreate();
        }
        const passID = c.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator.?);
        const nDependentDialects = 0;
        const dependentDialects = null;
        errdefer {
            e.enif_free_env(this.dispatcher_env);
        }
        // the env's ownership will be transferred to dispatcher to make sure string like op_name lives long enough
        const dispatcher: *CallbackDispatcher = CallbackDispatcher.init(this.dispatcher_env, this.pid) catch @panic("fail to init pass callback dispatcher");
        inline for (callback_names) |f| {
            if (this.cb_map.get(f)) |cb| {
                @field(dispatcher.callbacks, f) = e.enif_make_copy(dispatcher.env, cb);
            }
        }
        const pass = kinda.BangFunc(prelude.allKinds, c, "beaverPassCreate").wrap_ret_call(this.dispatcher_env, .{
            CallbackDispatcher.construct,
            CallbackDispatcher.destruct,
            CallbackDispatcher.initialize,
            CallbackDispatcher.clone,
            CallbackDispatcher.run,
            passID,
            string_ref.get_binary_as_string_ref(this.dispatcher_env, this.name) catch @panic("fail to string ref for name"),
            string_ref.get_binary_as_string_ref(this.dispatcher_env, this.argument) catch @panic("fail to string ref for argument"),
            string_ref.get_binary_as_string_ref(this.dispatcher_env, this.description) catch @panic("fail to string ref for description"),
            string_ref.get_binary_as_string_ref(this.dispatcher_env, this.op_name) catch @panic("fail to string ref for op_name"),
            nDependentDialects,
            dependentDialects,
            dispatcher,
        }) catch @panic("fail to create pass resource");
        if (!beam.send_advanced(null, this.pid, null, pass)) {
            @panic("fail to send pass");
        }
    }

    pub fn create_mlir_pass(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
        const creator = try PassCreator.init(try beam.self(env), args[1..5]);
        inline for (callback_names) |f| {
            const key = beam.make_atom(env, f);
            var cb: beam.term = undefined;
            if (e.enif_get_map_value(env, args[5], key, &cb) <= 0) {
                @panic("fail to get cb from map:" ++ f);
            }
            if (!beam.is_nil2(env, cb)) {
                try creator.cb_map.put(f, e.enif_make_copy(creator.dispatcher_env, cb));
            }
        }
        if (c.beaverContextAddWork(ctx, create_and_send_pass, @ptrCast(creator))) {
            return beam.make_atom(env, "async");
        } else {
            @panic("fail to add work to context to create pass");
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
    prelude.beaverRawNIF(PassCreator, "create_mlir_pass", 6),
    prelude.beaverRawNIF(ManagerRunner, "run_pm_on_op_async", 2),
    prelude.beaverRawNIF(ManagerRunner, "destroy_pm_async", 1),
};
