const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
const std = @import("std");
const e = @import("erl_nif");
const result = @import("kinda").result;
pub const c = @import("prelude.zig");
const mem = @import("std").mem;

pub fn Printer(comptime name: [*c]const u8, comptime ResourceKind: type, comptime print_fn: anytype) type {
    return struct {
        const Error = error{
            NullPointerFound,
        };
        const Buffer = std.ArrayList(u8);
        buffer: Buffer,
        fn collect_string_ref(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.C) void {
            const printer: *@This() = @ptrCast(@alignCast(userData));
            printer.*.buffer.appendSlice(s.data[0..s.length]) catch unreachable;
        }
        fn to_string(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const entity: ResourceKind.T = try ResourceKind.resource.fetch(env, args[0]);
            if (entity.ptr == null) {
                return Error.NullPointerFound;
            }
            var printer = @This(){ .buffer = Buffer.init(beam.allocator) };
            defer printer.buffer.deinit();
            print_fn(entity, collect_string_ref, &printer);
            return beam.make_slice(env, printer.buffer.items);
        }
        const entry = result.nif(name, 1, to_string).entry;
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

// memory layout {StringRef, real_binary, null}
fn beaver_raw_get_string_ref(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const Error = error{
        ResAllocFailure,
        NotBinary,
    };
    const StructT = mlir_capi.StringRef.T;
    const DataT = [*c]u8;
    var bin: beam.binary = undefined;
    if (0 == e.enif_inspect_binary(env, args[0], &bin)) {
        return Error.NotBinary;
    }
    const ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.StringRef.resource.t, @sizeOf(StructT) + bin.size + 1);
    if (ptr == null) {
        return Error.ResAllocFailure;
    }
    var dptr: DataT = @ptrCast(ptr);
    dptr += @sizeOf(StructT);
    const sptr: *StructT = @ptrCast(@alignCast(ptr));
    sptr.* = c.mlirStringRefCreate(dptr, bin.size);
    mem.copyForwards(u8, dptr[0..bin.size], bin.data[0..bin.size]);
    dptr[bin.size] = '\x00';
    return e.enif_make_resource(env, ptr);
}

pub const nifs = .{
    result.nif("beaver_raw_get_string_ref", 1, beaver_raw_get_string_ref).entry,
    result.nif("beaver_raw_string_ref_to_binary", 1, beaver_raw_string_ref_to_binary).entry,
    Printer("beaver_raw_to_string_attribute", mlir_capi.Attribute, c.mlirAttributePrint).entry,
    Printer("beaver_raw_to_string_type", mlir_capi.Type, c.mlirTypePrint).entry,
    Printer("beaver_raw_to_string_operation", mlir_capi.Operation, c.mlirOperationPrint).entry,
    Printer("beaver_raw_to_string_operation_specialized", mlir_capi.Operation, c.beaverOperationPrintSpecializedFrom).entry,
    Printer("beaver_raw_to_string_operation_generic", mlir_capi.Operation, c.beaverOperationPrintGenericOpForm).entry,
    Printer("beaver_raw_to_string_operation_bytecode", mlir_capi.Operation, c.mlirOperationWriteBytecode).entry,
    Printer("beaver_raw_to_string_value", mlir_capi.Value, c.mlirValuePrint).entry,
    Printer("beaver_raw_to_string_pm", mlir_capi.OpPassManager, c.mlirPrintPassPipeline).entry,
    Printer("beaver_raw_to_string_affine_map", mlir_capi.AffineMap, c.mlirAffineMapPrint).entry,
    Printer("beaver_raw_to_string_location", mlir_capi.Location, c.beaverLocationPrint).entry,
};
