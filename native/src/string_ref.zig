const mlir_capi = @import("mlir_capi.zig");
const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const result = kinda.result;
pub const c = @import("prelude.zig").c;
const mem = @import("std").mem;

pub fn get_binary_as_string_ref(env: beam.env, term: beam.term) !mlir_capi.StringRef.T {
    const bin = try beam.get_binary(env, term);
    return c.mlirStringRefCreate(bin.data, bin.size);
}

const print_nif_prefix = "beaver_raw_to_string_";
pub fn PrinterNIF(comptime ResourceKind: type, comptime print_fn: anytype) type {
    return struct {
        const Error = error{
            NullPointerFound,
        };
        const Buffer = std.array_list.Managed(u8);
        buffer: Buffer,
        fn collect_string_ref(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.c) void {
            const printer: *@This() = @ptrCast(@alignCast(userData));
            printer.*.buffer.appendSlice(s.data[0..s.length]) catch unreachable;
        }
        pub fn print_make(env: beam.env, entity: ResourceKind.T) !beam.term {
            if (entity.ptr == null) {
                return Error.NullPointerFound;
            }
            var printer = @This(){ .buffer = Buffer.init(beam.allocator) };
            defer printer.buffer.deinit();
            print_fn(entity, collect_string_ref, &printer);
            return beam.make_slice(env, printer.buffer.items);
        }
        fn to_string(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const entity: ResourceKind.T = try ResourceKind.resource.fetch(env, args[0]);
            return print_make(env, entity);
        }
        fn nif(comptime name: []const u8) e.ErlNifFunc {
            return result.nif(print_nif_prefix ++ name, 1, to_string).entry;
        }
    };
}

// collect multiple MLIR StringRef and join them as a single erlang binary
pub const Printer = struct {
    pub const ResourceKind = kinda.ResourceKind(@This(), "Elixir.Beaver.Printer");
    pub const PtrType = *@This();
    pub const ArrayType = [*]@This();
    const Error = error{ NullPointerFound, InvalidPrinter, @"Already flushed" };
    const Buffer = std.array_list.Managed(u8);
    const Flushed = std.atomic.Value(bool);
    buffer: Buffer,
    flushed: Flushed = Flushed.init(false),
    pub fn make(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
        const v = @This(){ .buffer = Buffer.init(beam.allocator) };
        return ResourceKind.resource.make(env, v) catch return beam.Error.@"Fail to create primitive";
    }
    pub const maker = .{ make, 0 };
    fn collect_string_ref(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.c) void {
        const printer: *@This() = @ptrCast(@alignCast(userData));
        printer.*.buffer.appendSlice(s.data[0..s.length]) catch unreachable;
    }
    pub fn destroy(_: beam.env, userData: ?*anyopaque) callconv(.c) void {
        const printer: *@This() = @ptrCast(@alignCast(userData));
        if (!printer.flushed.load(.acquire)) {
            printer.buffer.deinit();
        }
    }
    fn callback(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
        return try mlir_capi.StringCallback.resource.make(env, collect_string_ref);
    }
    pub fn flush(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const printer: *@This() = try ResourceKind.resource.fetch_ptr(env, args[0]);
        if (printer.flushed.load(.acquire)) return Error.@"Already flushed";
        defer printer.buffer.deinit(); // defer to free buffer after return term has been created
        printer.flushed.store(true, .release);
        return beam.make_slice(env, printer.buffer.items);
    }
    const entries = .{
        result.nif("beaver_raw_string_printer_callback", 0, callback).entry,
        result.nif("beaver_raw_string_printer_flush", 1, flush).entry,
    };
};

fn string_ref_to_binary(env: beam.env, s: mlir_capi.StringRef.T) beam.term {
    return beam.make_slice(env, s.data[0..s.length]);
}

// collect multiple MLIR StringRef as a list of erlang binary
pub const StringRefCollector = struct {
    const Container = std.array_list.Managed(beam.term);
    container: Container = undefined,
    env: beam.env = undefined,
    pub fn append(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.c) void {
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
    result.nif(print_nif_prefix ++ "StringRef", 1, beaver_raw_string_ref_to_binary).entry,
    PrinterNIF(mlir_capi.Attribute, c.mlirAttributePrint).nif("Attribute"),
    PrinterNIF(mlir_capi.Type, c.mlirTypePrint).nif("Type"),
    PrinterNIF(mlir_capi.Operation, c.mlirOperationPrint).nif("Operation"),
    PrinterNIF(mlir_capi.Operation, c.beaverOperationPrintSpecializedFrom).nif("OperationSpecialized"),
    PrinterNIF(mlir_capi.Operation, c.beaverOperationPrintGenericOpForm).nif("OperationGeneric"),
    PrinterNIF(mlir_capi.Operation, c.mlirOperationWriteBytecode).nif("OperationBytecode"),
    PrinterNIF(mlir_capi.Value, c.mlirValuePrint).nif("Value"),
    PrinterNIF(mlir_capi.OpPassManager, c.mlirPrintPassPipeline).nif("OpPassManager"),
    PrinterNIF(mlir_capi.AffineMap, c.mlirAffineMapPrint).nif("AffineMap"),
    PrinterNIF(mlir_capi.Location, c.beaverLocationPrint).nif("Location"),
    PrinterNIF(mlir_capi.Identifier, c.mlirIdentifierPrint).nif("Identifier"),
    PrinterNIF(mlir_capi.Diagnostic, c.mlirDiagnosticPrint).nif("Diagnostic"),
} ++ Printer.entries;
