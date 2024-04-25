const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
const std = @import("std");
const e = @import("erl_nif");
const result = @import("result.zig");
pub const c = @import("prelude.zig");
const mem = @import("std").mem;

pub fn Printer(comptime ResourceKind: type, comptime print_fn: anytype) type {
    return struct {
        const Buffer = std.ArrayList(u8);
        buffer: Buffer,
        fn collect_string_ref(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.C) void {
            var printer: *@This() = @ptrCast(@alignCast(userData));
            printer.*.buffer.appendSlice(s.data[0..s.length]) catch unreachable;
        }
        pub fn to_string(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            var entity: ResourceKind.T = ResourceKind.resource.fetch(env, args[0]) catch
                return beam.make_error_binary(env, "fail to fetch resource for MLIR entity to print, expected: " ++ @typeName(ResourceKind.T));
            if (entity.ptr == null) {
                return beam.make_error_binary(env, "null pointer found: " ++ @typeName(@TypeOf(entity)));
            }
            var printer = @This(){ .buffer = Buffer.init(beam.allocator) };
            defer printer.buffer.deinit();
            print_fn(entity, collect_string_ref, &printer);
            return beam.make_slice(env, printer.buffer.items);
        }
    };
}

fn string_ref_to_binary(env: beam.env, s: mlir_capi.StringRef.T) beam.term {
    return beam.make_slice(env, s.data[0..s.length]);
}

// collect mlir StringRef as a list of erlang binary
pub const StringRefCollector = struct {
    const Container = std.ArrayList(beam.term);
    container: Container = undefined,
    env: beam.env = undefined,
    pub fn append(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.C) void {
        const ptr: ?*@This() = @ptrCast(@alignCast(userData));
        if (ptr) |self| {
            self.container.append(string_ref_to_binary(self.env, s)) catch unreachable;
        }
    }
    pub fn init(env: beam.env) @This() {
        return @This(){ .container = Container.init(beam.allocator), .env = env };
    }
    pub fn collect_and_destroy(this: *@This()) !beam.term {
        defer this.container.deinit();
        return beam.make_term_list(this.env, this.container.items);
    }
};

fn beaver_raw_string_ref_to_binary(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return string_ref_to_binary(env, try mlir_capi.StringRef.resource.fetch(env, args[0]));
}

const Error = error{
    ResAllocFailure,
    NotBinary,
};

// memory layout {StringRef, real_binary, null}
fn beaver_raw_get_string_ref(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const StructT = mlir_capi.StringRef.T;
    const DataT = [*c]u8;
    var bin: beam.binary = undefined;
    if (0 == e.enif_inspect_binary(env, args[0], &bin)) {
        return Error.NotBinary;
    }
    var ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.StringRef.resource.t, @sizeOf(StructT) + bin.size + 1);
    if (ptr == null) {
        return Error.ResAllocFailure;
    }
    var dptr: DataT = @ptrCast(ptr);
    dptr += @sizeOf(StructT);
    var sptr: *StructT = @ptrCast(@alignCast(ptr));
    sptr.* = c.mlirStringRefCreate(dptr, bin.size);
    mem.copy(u8, dptr[0..bin.size], bin.data[0..bin.size]);
    dptr[bin.size] = '\x00';
    return e.enif_make_resource(env, ptr);
}

pub const nifs = .{ result.nif("beaver_raw_get_string_ref", 1, beaver_raw_get_string_ref).entry, result.nif("beaver_raw_string_ref_to_binary", 1, beaver_raw_string_ref_to_binary).entry };
