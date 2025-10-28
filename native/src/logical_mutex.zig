const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;

pub const Token = struct {
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    done: bool = false,
    logical_success: bool = false,
    caller_pid: beam.pid = undefined,
    pub var resource_type: beam.resource_type = undefined;
    pub const resource_name = "Beaver" ++ @typeName(@This());
    pub fn wait_logical(self: *@This()) struct { result: mlir_capi.LogicalResult.T, caller: beam.pid } {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (!self.done) {
            self.cond.wait(&self.mutex);
        }
        const res = if (self.logical_success) c.mlirLogicalResultSuccess() else c.mlirLogicalResultFailure();
        return .{ .result = res, .caller = self.caller_pid };
    }
    fn signal(self: *@This(), logical_success: bool, caller_pid: beam.pid) void {
        self.mutex.lock();
        if (self.done) {
            std.log.warn("Logical mutex token signaled more than once, will be a noop", .{});
            return;
        }
        defer self.mutex.unlock();
        self.logical_success = logical_success;
        self.done = true;
        self.caller_pid = caller_pid;
        self.cond.signal();
    }
    pub fn logical_mutex_signal(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        var token = try beam.fetch_ptr_resource_wrapped(@This(), env, args[0]);
        token.signal(try beam.get_bool(env, args[1]), try beam.self(env));
        return beam.make_ok(env);
    }
};

pub const nifs = .{prelude.beaverRawNIF(Token, "logical_mutex_signal", 2)};
pub fn open_all(env: beam.env) void {
    beam.open_resource_wrapped(env, Token);
}
