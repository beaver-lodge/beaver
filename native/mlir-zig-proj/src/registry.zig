const std = @import("std");
const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const debug_print = @import("std").debug.print;
const result = @import("result.zig");

fn context_of_dialects() mlir_capi.Context.T {
    const ctx = c.mlirContextCreate();
    var registry = c.mlirDialectRegistryCreate();
    c.mlirRegisterAllDialects(registry);
    c.mlirContextAppendDialectRegistry(ctx, registry);
    c.mlirDialectRegistryDestroy(registry);
    c.mlirContextLoadAllAvailableDialects(ctx);
    return ctx;
}

// collect mlir StringRef as a list of erlang binary
const StringRefCollector = struct {
    const Container = std.ArrayList(beam.term);
    container: Container = undefined,
    env: beam.env = undefined,
    fn append(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.C) void {
        const ptr: ?*@This() = @ptrCast(@alignCast(userData));
        if (ptr) |self| {
            self.container.append(beam.make_slice(self.env, s.data[0..s.length])) catch unreachable;
        }
    }
    fn init(env: beam.env) @This() {
        return @This(){ .container = Container.init(beam.allocator), .env = env };
    }
    fn collect(this: *@This()) !beam.term {
        defer this.container.deinit();
        return beam.make_term_list(this.env, try this.container.toOwnedSlice());
    }
};

fn get_all_registered_ops(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
    // we don't use beam.allocator.create here because MLIR will not free this user data
    var col = StringRefCollector.init(env);
    c.beaverGetRegisteredOps(ctx, StringRefCollector.append, @constCast(@ptrCast(@alignCast(&col))));
    return try col.collect();
}

fn get_registered_dialects(env: beam.env) !beam.term {
    const ctx = context_of_dialects();
    defer c.mlirContextDestroy(ctx);
    var num_dialects: usize = 0;
    // TODO: refactor this dirty trick
    var names: [300]mlir_capi.StringRef.T = undefined;
    c.beaverRegisteredDialects(ctx, &names, &num_dialects);
    if (num_dialects == 0) {
        return beam.make_error_binary(env, "no dialects found");
    }
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(num_dialects));
    defer beam.allocator.free(ret);
    var i: usize = 0;
    while (i < num_dialects) : ({
        i += 1;
    }) {
        ret[@intCast(i)] = beam.make_c_string_charlist(env, names[i].data);
    }
    return beam.make_term_list(env, ret);
}

pub const beaver_raw_registered_ops = result.nif("beaver_raw_registered_ops", 1, get_all_registered_ops).entry;

pub export fn beaver_raw_registered_dialects(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    return get_registered_dialects(env) catch beam.make_error_binary(env, "launching nif");
}
