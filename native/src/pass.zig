const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const callback_names = .{ "construct", "destruct", "initialize", "clone", "run" };
const RuntimeDispatcher = kinda.callback_runtime.Dispatcher(callback_names);
const string_ref = @import("string_ref.zig");
const callback_args = @import("callback_args.zig");

threadlocal var typeIDAllocator: ?mlir_capi.TypeIDAllocator.T = null;
const CallbackDispatcher = struct {
    fn construct(userData: ?*anyopaque) callconv(.c) void {
        const this: *RuntimeDispatcher = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const response = callback_args.invokeTerms(this, "construct", temp_env, .{}) catch unreachable;
        if (!response.success) @panic("Fail to construct a pass implemented in Elixir");
    }
    fn destruct(userData: ?*anyopaque) callconv(.c) void {
        const this: *RuntimeDispatcher = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const response = callback_args.invokeTerms(this, "destruct", temp_env, .{}) catch unreachable;
        if (!response.success) @panic("Fail to destruct a pass implemented in Elixir");
        this.deinit();
    }
    fn initialize(ctx: mlir_capi.Context.T, userData: ?*anyopaque) callconv(.c) mlir_capi.LogicalResult.T {
        const this: *RuntimeDispatcher = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const response = callback_args.invokeTerms(this, "initialize", temp_env, .{mlir_capi.Context.resource.make_kind(temp_env, ctx) catch unreachable}) catch return c.beaverLogicalResultFailure();
        if (!response.success) {
            const loc = c.mlirLocationUnknownGet(ctx);
            c.mlirEmitError(loc, "Fail to initialize a pass implemented in Elixir");
            return c.beaverLogicalResultFailure();
        }
        return c.beaverLogicalResultSuccess();
    }
    fn clone(userData: ?*anyopaque) callconv(.c) ?*anyopaque {
        const this: *RuntimeDispatcher = @ptrCast(@alignCast(userData));
        const new_dispatcher = (if (@hasDecl(RuntimeDispatcher, "cloneWithOwnership"))
            this.cloneWithOwnership(.bind_once)
        else
            this.clone()) catch unreachable;
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const response = callback_args.invokeTerms(new_dispatcher, "clone", temp_env, .{this.copyId(temp_env)}) catch unreachable;
        if (!response.success) @panic("Fail to clone a pass implemented in Elixir");
        return new_dispatcher;
    }
    fn run(op: mlir_capi.Operation.T, pass: c.MlirExternalPass, userData: ?*anyopaque) callconv(.c) void {
        const scope = c.beaverExecutionEnterOperation(op);
        defer c.beaverExecutionLeaveOperation(scope);
        const this: *RuntimeDispatcher = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const op_kind = mlir_capi.Operation.resource.make_kind(temp_env, op) catch return c.mlirExternalPassSignalFailure(pass);
        if (callback_args.invokeTerms(this, "run", temp_env, .{op_kind})) |response| {
            if (!response.success) {
                c.mlirEmitError(c.mlirOperationGetLocation(op), "Fail to run a pass implemented in Elixir");
                c.mlirExternalPassSignalFailure(pass);
            }
        } else |err| {
            c.mlirEmitError(c.mlirOperationGetLocation(op), @errorName(err));
            c.mlirExternalPassSignalFailure(pass);
        }
    }
};

pub fn create(env: beam.env, arguments: beam.term) !beam.term {
    const args = try @import("execution.zig").tuple(env, arguments, 5);
    var owner: beam.pid = undefined;
    if (e.enif_get_local_pid(env, args[4], &owner) == 0) return error.InvalidCallbackOwner;
    const dispatcher = try RuntimeDispatcher.initWithOptions(owner, callback_args.fixedOptions(RuntimeDispatcher, 30_000));
    errdefer dispatcher.deinit();
    inline for (callback_names) |name| dispatcher.setCallback(name, beam.make_atom(env, name));
    // MLIR borrows the operation anchor; keep its bytes with the callback owner.
    var names: [4]mlir_capi.StringRef.T = undefined;
    for (&names, args[0..4]) |*name, term| {
        name.* = try string_ref.get_binary_as_string_ref(dispatcher.env, e.enif_make_copy(dispatcher.env, term));
    }
    if (typeIDAllocator == null) typeIDAllocator = c.mlirTypeIDAllocatorCreate();
    return kinda.BangFunc(prelude.allKinds, c, "beaverPassCreate").wrap_ret_call(env, .{
        CallbackDispatcher.construct, CallbackDispatcher.destruct, CallbackDispatcher.initialize,
        CallbackDispatcher.clone,     CallbackDispatcher.run,      c.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator.?),
        names[0],                     names[1],                    names[2],
        names[3],                     0,                           null,
        dispatcher,
    });
}

var mutex: std.Io.Mutex = .init;
var all_passes_registered = false;
pub fn register_all_passes() void {
    const io = std.Options.debug_io;
    mutex.lockUncancelable(io);
    defer mutex.unlock(io);
    if (!all_passes_registered) {
        all_passes_registered = true;
        c.mlirRegisterAllPasses();
    }
}
