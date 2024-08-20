const std = @import("std");
const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const debug_print = @import("std").debug.print;
const result = @import("kinda").result;
const diagnostic = @import("diagnostic.zig");

const Token = struct {
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    done: bool = false,
    pub var resource_type: beam.resource_type = undefined;
    pub const resource_name = "Beaver" ++ @typeName(@This());
    fn wait(self: *@This()) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (!self.done) {
            self.cond.wait(&self.mutex);
        }
    }
    fn signal(self: *@This()) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.done = true;
        self.cond.signal();
    }
    fn pass_token_signal(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        var token = try beam.fetch_ptr_resource_wrapped(@This(), env, args[0]);
        token.signal();
        return beam.make_ok(env);
    }
};

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
    const Error = error{
        EnvAllocFailure,
        MsgSendFailure,
    };
    fn do_run(op: mlir_capi.Operation.T, pass: c.MlirExternalPass, userData: ?*anyopaque) !void {
        const ud: *@This() = @ptrCast(@alignCast(userData));
        const env = e.enif_alloc_env() orelse return Error.EnvAllocFailure;
        defer e.enif_clear_env(env);
        const handler = ud.*.handler;
        var tuple_slice: []beam.term = try beam.allocator.alloc(beam.term, 4);
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_atom(env, "run");
        tuple_slice[1] = try mlir_capi.Operation.resource.make(env, op);
        tuple_slice[2] = try mlir_capi.ExternalPass.resource.make(env, pass);
        var token = Token{};
        tuple_slice[3] = try beam.make_ptr_resource_wrapped(env, &token);
        if (!beam.send(env, handler, beam.make_tuple(env, tuple_slice))) {
            return Error.MsgSendFailure;
        }
        token.wait();
    }
    fn run(op: mlir_capi.Operation.T, pass: c.MlirExternalPass, userData: ?*anyopaque) callconv(.C) void {
        if (do_run(op, pass, userData)) |_| {} else |err| {
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

pub const nifs = .{ result.nif("beaver_raw_create_mlir_pass", 5, do_create).entry, result.nif("beaver_raw_pass_token_signal", 1, Token.pass_token_signal).entry };
pub fn open_all(env: beam.env) void {
    beam.open_resource_wrapped(env, Token);
}
