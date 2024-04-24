const std = @import("std");
const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const debug_print = @import("std").debug.print;
const result = @import("result.zig");

pub const Token = struct {
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
    const callbacks: mlir_capi.ExternalPassCallbacks.T = mlir_capi.ExternalPassCallbacks.T{
        .construct = construct,
        .destruct = destruct,
        .initialize = initialize,
        .clone = clone,
        .run = run,
    };
    handler: beam.pid,
    fn construct(_: ?*anyopaque) callconv(.C) void {}

    fn destruct(userData: ?*anyopaque) callconv(.C) void {
        const ptr: *@This() = @ptrCast(@alignCast(userData));
        beam.allocator.destroy(ptr);
    }
    fn initialize(_: mlir_capi.Context.T, _: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        return mlir_capi.LogicalResult.T{ .value = 1 };
    }
    fn clone(userData: ?*anyopaque) callconv(.C) ?*anyopaque {
        const old: *@This() = @ptrCast(@alignCast(userData));
        var new = beam.allocator.create(@This()) catch unreachable;
        new.* = old.*;
        return new;
    }
    fn run(op: mlir_capi.Operation.T, pass: c.MlirExternalPass, userData: ?*anyopaque) callconv(.C) void {
        const ud: *@This() = @ptrCast(@alignCast(userData));
        const env = e.enif_alloc_env() orelse {
            debug_print("fail to creat env\n", .{});
            return c.mlirExternalPassSignalFailure(pass);
        };
        defer e.enif_clear_env(env);
        const handler = ud.*.handler;
        var tuple_slice: []beam.term = beam.allocator.alloc(beam.term, 4) catch unreachable;
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_atom(env, "run");
        tuple_slice[1] = beam.make_resource(env, op, mlir_capi.Operation.resource.t) catch {
            debug_print("fail to make res: {}\n", .{@TypeOf(op)});
            unreachable;
        };
        tuple_slice[2] = beam.make_resource(env, pass, mlir_capi.ExternalPass.resource.t) catch {
            debug_print("fail to make res: {}\n", .{@TypeOf(pass)});
            unreachable;
        };
        var token = Token{};
        tuple_slice[3] = beam.make_ptr_resource_wrapped(env, &token) catch {
            unreachable;
        };
        if (!beam.send(env, handler, beam.make_tuple(env, tuple_slice))) {
            debug_print("fail to send message to pass handler.\n", .{});
            c.mlirExternalPassSignalFailure(pass);
        }
        token.wait();
    }
};

pub fn do_create(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const name: mlir_capi.StringRef.T = try beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[0]);
    const argument: mlir_capi.StringRef.T = try beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[1]);
    const description: mlir_capi.StringRef.T = try beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[2]);
    const op_name: mlir_capi.StringRef.T = try beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[3]);
    const handler: beam.pid = try beam.get_pid(env, args[4]);

    const typeIDAllocator = c.mlirTypeIDAllocatorCreate();
    defer c.mlirTypeIDAllocatorDestroy(typeIDAllocator);
    const passID = c.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    const nDependentDialects = 0;
    const dependentDialects = null;
    var bp: *BeaverPass = try beam.allocator.create(BeaverPass);
    bp.* = BeaverPass{ .handler = handler };
    // use this function to avoid ABI issue
    const ep = c.beaverCreateExternalPass(
        passID,
        name,
        argument,
        description,
        op_name,
        nDependentDialects,
        dependentDialects,
        BeaverPass.construct,
        BeaverPass.destruct,
        BeaverPass.initialize,
        BeaverPass.clone,
        BeaverPass.run,
        bp,
    );
    return try mlir_capi.Pass.resource.make(env, ep);
}
const create = result.nif("beaver_raw_create_mlir_pass", 5, do_create).entry;
const token_signal = result.nif("beaver_raw_pass_token_signal", 1, Token.pass_token_signal).entry;
pub const nifs = .{ create, token_signal };
