const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
const std = @import("std");
const e = @import("erl_nif");
const result = @import("kinda").result;
const kinda = @import("kinda");
pub const c = @import("prelude.zig");
const mem = @import("std").mem;

pub fn PrinterNIF(comptime name: []const u8, comptime ResourceKind: type, comptime print_fn: anytype) type {
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
        const entry = result.nif("beaver_raw_to_string_" ++ name, 1, to_string).entry;
    };
}

// collect multiple MLIR StringRef and join them as a single erlang binary
pub const Printer = struct {
    pub const ResourceKind = kinda.ResourceKind(@This(), "Elixir.Beaver.StringPrinter");
    pub const PtrType = *@This();
    pub const ArrayType = [*]@This();
    const Error = error{ NullPointerFound, InvalidPrinter, @"Already flushed" };
    const Buffer = std.ArrayList(u8);
    buffer: Buffer,
    flushed: bool = false,
    pub fn make(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
        const v = @This(){ .buffer = Buffer.init(beam.allocator) };
        return ResourceKind.resource.make(env, v) catch return beam.Error.@"Fail to create primitive";
    }
    pub const maker = .{ make, 0 };
    fn collect_string_ref(s: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.C) void {
        const printer: *@This() = @ptrCast(@alignCast(userData));
        printer.*.buffer.appendSlice(s.data[0..s.length]) catch unreachable;
    }
    pub fn destroy(_: beam.env, userData: ?*anyopaque) callconv(.C) void {
        const printer: *@This() = @ptrCast(@alignCast(userData));
        if (!printer.flushed) {
            printer.buffer.deinit();
        }
    }
    fn callback(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
        return try mlir_capi.StringCallback.resource.make(env, collect_string_ref);
    }
    pub fn flush(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const printer: *@This() = try ResourceKind.resource.fetch_ptr(env, args[0]);
        if (printer.flushed) return Error.@"Already flushed";
        printer.flushed = true;
        printer.buffer.deinit();
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
    PrinterNIF("Attribute", mlir_capi.Attribute, c.mlirAttributePrint).entry,
    PrinterNIF("Type", mlir_capi.Type, c.mlirTypePrint).entry,
    PrinterNIF("Operation", mlir_capi.Operation, c.mlirOperationPrint).entry,
    PrinterNIF("OperationSpecialized", mlir_capi.Operation, c.beaverOperationPrintSpecializedFrom).entry,
    PrinterNIF("OperationGeneric", mlir_capi.Operation, c.beaverOperationPrintGenericOpForm).entry,
    PrinterNIF("OperationBytecode", mlir_capi.Operation, c.mlirOperationWriteBytecode).entry,
    PrinterNIF("Value", mlir_capi.Value, c.mlirValuePrint).entry,
    PrinterNIF("OpPassManager", mlir_capi.OpPassManager, c.mlirPrintPassPipeline).entry,
    PrinterNIF("AffineMap", mlir_capi.AffineMap, c.mlirAffineMapPrint).entry,
    PrinterNIF("Location", mlir_capi.Location, c.beaverLocationPrint).entry,
    PrinterNIF("Identifier", mlir_capi.Identifier, c.mlirIdentifierPrint).entry,
    PrinterNIF("Diagnostic", mlir_capi.Diagnostic, c.mlirDiagnosticPrint).entry,
} ++ Printer.entries;
