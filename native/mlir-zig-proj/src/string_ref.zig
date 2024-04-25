const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
const std = @import("std");

// collect mlir StringRef as a list of erlang binary
pub const StringRefCollector = struct {
    const Container = std.ArrayList(beam.term);
    container: Container = undefined,
    env: beam.env = undefined,
    pub fn append(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.C) void {
        const ptr: ?*@This() = @ptrCast(@alignCast(userData));
        if (ptr) |self| {
            self.container.append(beam.make_slice(self.env, s.data[0..s.length])) catch unreachable;
        }
    }
    pub fn init(env: beam.env) @This() {
        return @This(){ .container = Container.init(beam.allocator), .env = env };
    }
    pub fn collect_and_destroy(this: *@This()) !beam.term {
        defer this.container.deinit();
        return beam.make_term_list(this.env, try this.container.toOwnedSlice());
    }
};
