const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const beam = @import("beam");
const result = @import("kinda").result;
const e = @import("erl_nif");

pub const Token = struct {
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    done: bool = false,
    logical_success: bool = false,
    pub var resource_type: beam.resource_type = undefined;
    pub const resource_name = "Beaver" ++ @typeName(@This());
    fn wait(self: *@This()) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (!self.done) {
            self.cond.wait(&self.mutex);
        }
    }
    pub fn wait_logical(self: *@This()) mlir_capi.LogicalResult.T {
        wait(self);
        return if (self.logical_success) c.mlirLogicalResultSuccess() else c.mlirLogicalResultFailure();
    }
    fn signal(self: *@This(), logical_success: bool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.logical_success = logical_success;
        self.done = true;
        self.cond.signal();
    }
    pub fn signal_logical_success(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        var token = try beam.fetch_ptr_resource_wrapped(@This(), env, args[0]);
        token.signal(beam.get_bool(env, args[1]) catch unreachable);
        return beam.make_ok(env);
    }
};

pub const nifs = .{
    result.nif("beaver_raw_logical_mutex_token_signal_success", 2, Token.signal_logical_success).entry,
};
pub fn open_all(env: beam.env) void {
    beam.open_resource_wrapped(env, Token);
}
