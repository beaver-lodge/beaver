const std = @import("std");
const testing = std.testing;

export fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try testing.expect(add(3, 7) == 10);
}
const beam = @import("beam.zig");
const kinda = @import("kinda.zig");
const e = @import("erl_nif.zig");
const mlir_capi = @import("mlir.imp.zig");
pub const c = mlir_capi.c;

pub fn make_charlist_from_string_ref(environment: beam.env, val: c.struct_MlirStringRef) beam.term {
    return beam.make_charlist_len(environment, val.data, val.length);
}

fn get_all_registered_ops(env: beam.env) !beam.term {
    const ctx = get_context_load_all_dialects();
    defer c.mlirContextDestroy(ctx);
    const num_op: isize = c.beaverGetNumRegisteredOperations(ctx);
    var i: isize = 0;
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(usize, num_op));
    defer beam.allocator.free(ret);
    while (i < num_op) : ({
        i += 1;
    }) {
        const registered_op_name = c.beaverGetRegisteredOperationName(ctx, i);
        const dialect_name = c.beaverRegisteredOperationNameGetDialectName(registered_op_name);
        const op_name = c.beaverRegisteredOperationNameGetOpName(registered_op_name);
        var tuple_slice: []beam.term = try beam.allocator.alloc(beam.term, 2);
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = make_charlist_from_string_ref(env, dialect_name);
        tuple_slice[1] = make_charlist_from_string_ref(env, op_name);
        ret[@intCast(usize, i)] = beam.make_tuple(env, tuple_slice);
    }
    return beam.make_term_list(env, ret);
}

fn get_context_load_all_dialects() c.struct_MlirContext {
    const ctx = c.mlirContextCreate();
    var registry = c.mlirDialectRegistryCreate();
    c.mlirRegisterAllDialects(registry);
    c.mlirContextAppendDialectRegistry(ctx, registry);
    c.mlirDialectRegistryDestroy(registry);
    c.mlirContextLoadAllAvailableDialects(ctx);
    return ctx;
}

fn get_all_registered_ops2(env: beam.env, dialect: c.struct_MlirStringRef) !beam.term {
    const ctx = get_context_load_all_dialects();
    defer c.mlirContextDestroy(ctx);
    var num_op: usize = 0;
    // TODO: refactor this dirty trick
    var names: [300]c.struct_MlirRegisteredOperationName = undefined;
    c.beaverRegisteredOperationsOfDialect(ctx, dialect, &names, &num_op);
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(usize, num_op));
    defer beam.allocator.free(ret);
    var i: usize = 0;
    while (i < num_op) : ({
        i += 1;
    }) {
        const registered_op_name = names[i];
        const op_name = c.beaverRegisteredOperationNameGetOpName(registered_op_name);
        ret[@intCast(usize, i)] = beam.make_c_string_charlist(env, op_name.data);
    }
    return beam.make_term_list(env, ret);
}

export fn beaver_raw_registered_ops(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    return get_all_registered_ops(env) catch beam.make_error_binary(env, "launching nif");
}

fn get_registered_dialects(env: beam.env) !beam.term {
    const ctx = get_context_load_all_dialects();
    defer c.mlirContextDestroy(ctx);
    var num_dialects: usize = 0;
    // TODO: refactor this dirty trick
    var names: [300]c.struct_MlirStringRef = undefined;
    c.beaverRegisteredDialects(ctx, &names, &num_dialects);
    if (num_dialects == 0) {
        return beam.make_error_binary(env, "no dialects found");
    }
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(usize, num_dialects));
    defer beam.allocator.free(ret);
    var i: usize = 0;
    while (i < num_dialects) : ({
        i += 1;
    }) {
        ret[@intCast(usize, i)] = beam.make_c_string_charlist(env, names[i].data);
    }
    return beam.make_term_list(env, ret);
}

export fn beaver_raw_registered_ops_of_dialect(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var dialect: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, mlir_capi.MlirStringRef.resource.t, args[0])) |value| {
        dialect = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for dialect, expected: c.struct_MlirStringRef");
    }
    return get_all_registered_ops2(env, dialect) catch beam.make_error_binary(env, "launching nif");
}

export fn beaver_raw_registered_dialects(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    return get_registered_dialects(env) catch beam.make_error_binary(env, "launching nif");
}

export fn beaver_raw_resource_c_string_to_term_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: mlir_capi.U8.Array.T = beam.fetch_resource(mlir_capi.U8.Array.T, env, mlir_capi.U8.Array.resource.t, args[0]) catch return beam.make_error_binary(env, "fail to fetch resource of c string");
    return beam.make_c_string_charlist(env, arg0);
}

// create a C string resource by copying given binary
const mem = @import("std").mem;
// memory layout {address, real_binary, null}
export fn beaver_raw_get_resource_c_string(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    const RType = [*c]u8;
    var bin: beam.binary = undefined;
    if (0 == e.enif_inspect_binary(env, args[0], &bin)) {
        return beam.make_error_binary(env, "not a binary");
    }
    var ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.U8.Array.resource.t, @sizeOf(RType) + bin.size + 1);
    var obj: *RType = undefined;
    var real_binary: RType = undefined;
    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
        real_binary = @ptrCast(RType, ptr);
        real_binary += @sizeOf(RType);
        real_binary[bin.size] = 0;
        obj.* = real_binary;
    }
    mem.copy(u8, real_binary[0..bin.size], bin.data[0..bin.size]);
    return e.enif_make_resource(env, ptr);
}

export fn beaver_raw_mlir_named_attribute_get(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: c.struct_MlirIdentifier = undefined;
    if (beam.fetch_resource(c.struct_MlirIdentifier, env, mlir_capi.MlirIdentifier.resource.t, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.struct_MlirIdentifier");
    }
    var arg1: c.struct_MlirAttribute = undefined;
    if (beam.fetch_resource(c.struct_MlirAttribute, env, mlir_capi.MlirAttribute.resource.t, args[1])) |value| {
        arg1 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #2, expected: c.struct_MlirAttribute");
    }

    var ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.MlirNamedAttribute.resource.t, @sizeOf(c.struct_MlirNamedAttribute));

    const RType = c.struct_MlirNamedAttribute;
    var obj: *RType = undefined;

    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
        obj.* = c.struct_MlirNamedAttribute{ .name = arg0, .attribute = arg1 };
    }
    return e.enif_make_resource(env, ptr);
}

const StringRefCollector = struct { env: beam.env, list: std.ArrayList(beam.term) };

fn collect_string_ref(string_ref: c.struct_MlirStringRef, collector: ?*anyopaque) callconv(.C) void {
    var collector_ptr = @ptrCast(*StringRefCollector, @alignCast(@alignOf(*StringRefCollector), collector));
    collector_ptr.*.list.append(make_charlist_from_string_ref(collector_ptr.*.env, string_ref)) catch unreachable;
}

fn print_mlir(env: beam.env, element: anytype, printer: anytype) beam.term {
    var list = std.ArrayList(beam.term).init(beam.allocator);
    var collector = StringRefCollector{ .env = env, .list = list };
    defer list.deinit();
    printer(element, collect_string_ref, &collector);
    return beam.make_term_list(env, collector.list.items);
}

export fn beaver_raw_beaver_attribute_to_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: c.struct_MlirAttribute = undefined;
    if (beam.fetch_resource(c.struct_MlirAttribute, env, mlir_capi.MlirAttribute.resource.t, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.struct_MlirAttribute");
    }
    return print_mlir(env, arg0, c.mlirAttributePrint);
}

export fn beaver_raw_beaver_type_to_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: c.struct_MlirType = undefined;
    if (beam.fetch_resource(c.struct_MlirType, env, mlir_capi.MlirType.resource.t, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.struct_MlirType");
    }
    return print_mlir(env, arg0, c.mlirTypePrint);
}

export fn beaver_raw_beaver_operation_to_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: c.struct_MlirOperation = undefined;
    if (beam.fetch_resource(c.struct_MlirOperation, env, mlir_capi.MlirOperation.resource.t, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.struct_MlirOperation");
    }
    return print_mlir(env, arg0, c.mlirOperationPrint);
}

export fn beaver_raw_get_context_load_all_dialects(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    var ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.MlirContext.resource.t, @sizeOf(c.struct_MlirContext));
    const RType = c.struct_MlirContext;
    var obj: *RType = undefined;
    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
        obj.* = get_context_load_all_dialects();
    }
    return e.enif_make_resource(env, ptr);
}

const PassToken = struct {
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
    export fn pass_token_signal(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
        var token = beam.fetch_ptr_resource_wrapped(@This(), env, args[0]) catch return beam.make_error_binary(env, "fail to fetch resource for pass token");
        token.signal();
        return beam.make_ok(env);
    }
};

const print = @import("std").debug.print;
const BeaverPass = struct {
    const UserData = struct { handler: beam.pid };
    fn construct(_: ?*anyopaque) callconv(.C) void {}

    fn destruct(userData: ?*anyopaque) callconv(.C) void {
        const ptr = @ptrCast(*UserData, @alignCast(@alignOf(UserData), userData));
        beam.allocator.destroy(ptr);
    }
    fn initialize(_: c.struct_MlirContext, _: ?*anyopaque) callconv(.C) c.struct_MlirLogicalResult {
        return c.struct_MlirLogicalResult{ .value = 1 };
    }
    fn clone(userData: ?*anyopaque) callconv(.C) ?*anyopaque {
        const old = @ptrCast(*UserData, @alignCast(@alignOf(*UserData), userData));
        var new = beam.allocator.create(UserData) catch unreachable;
        new.* = old.*;
        return new;
    }
    fn run(op: c.struct_MlirOperation, pass: c.struct_MlirExternalPass, userData: ?*anyopaque) callconv(.C) void {
        if (1 > 2) {
            c.mlirExternalPassSignalFailure(pass);
        }
        const ud = @ptrCast(*UserData, @alignCast(@alignOf(UserData), userData));
        const env = e.enif_alloc_env() orelse {
            print("fail to creat env\n", .{});
            return c.mlirExternalPassSignalFailure(pass);
        };
        defer e.enif_clear_env(env);
        const handler = ud.*.handler;
        var tuple_slice: []beam.term = beam.allocator.alloc(beam.term, 4) catch unreachable;
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_atom(env, "run");
        tuple_slice[1] = beam.make_resource(env, op, mlir_capi.MlirOperation.resource.t) catch {
            print("fail to make res: {}\n", .{@TypeOf(op)});
            unreachable;
        };
        tuple_slice[2] = beam.make_resource(env, pass, mlir_capi.MlirExternalPass.resource.t) catch {
            print("fail to make res: {}\n", .{@TypeOf(pass)});
            unreachable;
        };
        var token = PassToken{};
        tuple_slice[3] = beam.make_ptr_resource_wrapped(env, &token) catch {
            unreachable;
        };
        if (!beam.send(env, handler, beam.make_tuple(env, tuple_slice))) {
            print("fail to send message to pass handler.\n", .{});
            c.mlirExternalPassSignalFailure(pass);
        }
        token.wait();
    }
};

export fn beaver_raw_read_opaque_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var ptr: mlir_capi.OpaquePtr.T = mlir_capi.OpaquePtr.resource.fetch(env, args[0]) catch
        return beam.make_error_binary(env, "fail to fetch resource for ptr, expected: " ++ @typeName(mlir_capi.OpaquePtr.T));
    var len = mlir_capi.U64.resource.fetch(env, args[1]) catch return beam.make_error_binary(env, "fail to fetch resource length, expected a integer");
    if (ptr == null) {
        return beam.make_error_binary(env, "ptr is null");
    }
    const slice = @ptrCast(mlir_capi.U8.Array.T, ptr)[0..len];
    return beam.make_slice(env, slice);
}

export fn beaver_raw_create_mlir_pass(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var name: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, mlir_capi.MlirStringRef.resource.t, args[0])) |value| {
        name = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass name, expected: c.struct_MlirStringRef");
    }
    var argument: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, mlir_capi.MlirStringRef.resource.t, args[1])) |value| {
        argument = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass argument, expected: c.struct_MlirStringRef");
    }
    var description: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, mlir_capi.MlirStringRef.resource.t, args[2])) |value| {
        description = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass description, expected: c.struct_MlirStringRef");
    }
    var op_name: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, mlir_capi.MlirStringRef.resource.t, args[3])) |value| {
        op_name = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass op name, expected: c.struct_MlirStringRef");
    }
    var handler: beam.pid = beam.get_pid(env, args[4]) catch return beam.make_error_binary(env, "expect the handler to be a pid");

    const typeIDAllocator = c.mlirTypeIDAllocatorCreate();
    defer c.mlirTypeIDAllocatorDestroy(typeIDAllocator);
    const passID = c.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    const nDependentDialects = 0;
    const dependentDialects = 0;
    var userData: *BeaverPass.UserData = beam.allocator.create(BeaverPass.UserData) catch return beam.make_error_binary(env, "fail to allocate for pass userdata");
    userData.*.handler = handler;
    const RType = c.struct_MlirPass;
    var ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.Pass.resource.t, @sizeOf(RType));
    var obj: *RType = undefined;
    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
        obj.* = c.beaverCreateExternalPass(
            BeaverPass.construct, // move it first to prevent calling wrong function
            passID,
            name,
            argument,
            description,
            op_name,
            nDependentDialects,
            dependentDialects,
            BeaverPass.destruct,
            BeaverPass.initialize,
            BeaverPass.clone,
            BeaverPass.run,
            userData,
        );
    }
    return e.enif_make_resource(env, ptr);
}

fn UnrankMemRefDescriptor(comptime ResourceKind: type) type {
    return extern struct {
        pub fn make(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            var allocated: ResourceKind.Ptr.T = ResourceKind.Ptr.resource.fetch(env, args[0]) catch
                return beam.make_error_binary(env, "fail to fetch allocated. expected: " ++ @typeName(ResourceKind.Ptr.T));
            var aligned: ResourceKind.Ptr.T = ResourceKind.Ptr.resource.fetch(env, args[1]) catch
                return beam.make_error_binary(env, "fail to fetch aligned. expected: " ++ @typeName(ResourceKind.Ptr.T));
            var offset: mlir_capi.I64.T = mlir_capi.I64.resource.fetch(env, args[2]) catch
                return beam.make_error_binary(env, "fail to fetch offset");
            const kind: type = dataKindToMemrefKind(ResourceKind);

            // unrank has different type, so put it in a dedicated arm
            var descriptor: UnrankMemRefDescriptor(ResourceKind) = undefined;
            // TODO: figure out how to write this in a more elegant way
            if (allocated == null) {
                descriptor = .{
                    .offset = offset,
                };
            } else {
                descriptor = .{
                    .allocated = allocated,
                    .aligned = aligned,
                    .offset = offset,
                };
            }

            return kind.per_rank_resource_kinds[0].resource.make(env, descriptor) catch return beam.make_error_binary(env, "fail to make unranked memref descriptor");
        }
        pub const maker = .{ make, 5 };
        const T = ResourceKind.T;
        allocated: ?*T = null,
        aligned: ?*T = null,
        offset: i64 = undefined,
    };
}

fn MemRefDescriptor(comptime ResourceKind: type, comptime N: usize) type {
    return extern struct {
        const T = ResourceKind.T;
        allocated: ?*T = null,
        aligned: ?*T = null,
        offset: i64 = undefined,
        sizes: [N]i64 = undefined,
        strides: [N]i64 = undefined,
        fn populate(self: *@This(), allocated: *T, aligned: *T, offset: i64, sizes: []i64, strides: []i64) void {
            self.allocated = allocated;
            self.aligned = aligned;
            self.offset = offset;
            const rank = sizes.len;
            mem.copy(i64, self.sizes[0..rank], sizes[0..rank]);
            mem.copy(i64, self.strides[0..rank], strides[0..rank]);
        }
        fn populate2(self: *@This(), offset: i64, sizes: []i64, strides: []i64) void {
            self.offset = offset;
            const rank = sizes.len;
            mem.copy(i64, self.sizes[0..rank], sizes[0..rank]);
            mem.copy(i64, self.strides[0..rank], strides[0..rank]);
        }
        pub fn make(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            var allocated: ResourceKind.Ptr.T = ResourceKind.Ptr.resource.fetch(env, args[0]) catch
                return beam.make_error_binary(env, "fail to fetch allocated. expected: " ++ @typeName(ResourceKind.Ptr.T));
            var aligned: ResourceKind.Ptr.T = ResourceKind.Ptr.resource.fetch(env, args[1]) catch
                return beam.make_error_binary(env, "fail to fetch aligned. expected: " ++ @typeName(ResourceKind.Ptr.T));
            var offset: mlir_capi.I64.T = mlir_capi.I64.resource.fetch(env, args[2]) catch
                return beam.make_error_binary(env, "fail to fetch offset");
            const sizes = beam.get_slice_of(i64, env, args[3]) catch return beam.make_error_binary(env, "fail to get sizes as zig slice");
            defer beam.allocator.free(sizes);
            const strides = beam.get_slice_of(i64, env, args[4]) catch return beam.make_error_binary(env, "fail to get sizes as zig slice");
            defer beam.allocator.free(strides);
            if (sizes.len != strides.len) {
                return beam.make_error_binary(env, "sizes and strides must have the same length");
            }
            const kind: type = dataKindToMemrefKind(ResourceKind);
            comptime var rank = N;
            if (rank != sizes.len) {
                return beam.make_error_binary(env, "wrong sizes for " ++ @typeName(@This()));
            }
            if (rank != strides.len) {
                return beam.make_error_binary(env, "wrong strides for " ++ @typeName(@This()));
            }
            var descriptor: MemRefDescriptor(ResourceKind, rank) = .{};
            if (allocated == null) {
                descriptor.populate2(offset, sizes, strides);
            } else {
                descriptor.populate(allocated, aligned, offset, sizes, strides);
            }
            return kind.per_rank_resource_kinds[rank].resource.make(env, descriptor) catch return beam.make_error_binary(env, "fail to make memref descriptor");
        }
        pub const maker = .{ make, 5 };
    };
}

const forward_module = "Elixir.Beaver.Native.Complex.F32";
const Complex = struct {
    fn of(comptime ElementKind: type) type {
        return struct {
            const T = extern struct {
                i: ElementKind.T,
                r: ElementKind.T,
            };
        };
    }
    const F32 = kinda.ResourceKind(Complex.of(mlir_capi.F32).T, forward_module);
};

const MemRefDataType = enum {
    @"Complex.F32",
    U8,
    U16,
    U32,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
};

fn dataTypeToResourceKind(self: MemRefDataType) type {
    return switch (self) {
        .@"Complex.F32" => Complex.F32,
        .U8 => mlir_capi.U8,
        .U16 => mlir_capi.U16,
        .U32 => mlir_capi.U32,
        .F32 => mlir_capi.F32,
        .F64 => mlir_capi.F64,
        .I8 => mlir_capi.I8,
        .I16 => mlir_capi.I16,
        .I32 => mlir_capi.I32,
        .I64 => mlir_capi.I64,
    };
}

fn dataKindToDataType(comptime self: type) MemRefDataType {
    return switch (self) {
        Complex.F32 => MemRefDataType.@"Complex.F32",
        mlir_capi.U8 => MemRefDataType.U8,
        mlir_capi.U16 => MemRefDataType.U16,
        mlir_capi.U32 => MemRefDataType.U32,
        mlir_capi.F32 => MemRefDataType.F32,
        mlir_capi.F64 => MemRefDataType.F64,
        mlir_capi.I8 => MemRefDataType.I8,
        mlir_capi.I16 => MemRefDataType.I16,
        mlir_capi.I32 => MemRefDataType.I32,
        mlir_capi.I64 => MemRefDataType.I64,
        else => unreachable(),
    };
}

const memref_kinds = .{
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.@"Complex.F32")),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.U8)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.U16)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.U32)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.F32)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.F64)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.I8)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.I16)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.I32)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.I64)),
};

fn dataKindToMemrefKind(comptime self: type) type {
    const dt = dataKindToDataType(self);
    const index = @enumToInt(dt);
    return memref_kinds[index];
}

const MemRefRankType = enum {
    DescriptorUnranked,
    Descriptor1D,
    Descriptor2D,
    Descriptor3D,
    Descriptor4D,
    Descriptor5D,
    Descriptor6D,
    Descriptor7D,
    Descriptor8D,
    Descriptor9D,
};

fn BeaverMemRef(comptime ResourceKind: type) type {
    return struct {
        fn aligned_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            const kind: type = dataKindToMemrefKind(ResourceKind);
            comptime var rank = 0;
            inline while (rank < kind.per_rank_resource_kinds.len) : (rank += 1) {
                if (kind.per_rank_resource_kinds[rank].resource.fetch(env, args[0])) |descriptor| {
                    var ret: mlir_capi.OpaquePtr.T = @ptrCast(mlir_capi.OpaquePtr.T, descriptor.aligned);
                    return mlir_capi.OpaquePtr.resource.make(env, ret) catch return beam.make_error_binary(env, "fail to make aligned ptr");
                } else |_| {
                    // do nothing
                }
            }
            return beam.make_error_binary(env, "fail to get aligned ptr");
        }
        fn self_opaque_ptr(env: beam.env, num_args: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            const kind: type = dataKindToMemrefKind(ResourceKind);
            comptime var rank = 0;
            inline while (rank < kind.per_rank_resource_kinds.len) : (rank += 1) {
                const per_rank_kind = kind.per_rank_resource_kinds[rank];
                if (per_rank_kind.resource.fetch(env, args[0])) |_| {
                    return per_rank_kind.opaque_ptr(env, num_args, args);
                } else |_| {
                    // do nothing
                }
            }
            return beam.make_error_binary(env, "fail to get opaque ptr to memref descriptor");
        }
        fn memref_dump(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            const kind: type = dataKindToMemrefKind(ResourceKind);
            comptime var rank = 0;
            inline while (rank < kind.per_rank_resource_kinds.len) : (rank += 1) {
                const per_rank_kind = kind.per_rank_resource_kinds[rank];
                if (per_rank_kind.resource.fetch(env, args[0])) |v| {
                    print("{}\n", .{v});
                    return beam.make_ok(env);
                } else |_| {
                    // do nothing
                }
            }
            return beam.make_error_binary(env, "fail to get opaque ptr to memref descriptor");
        }
        const data_kind_name = ResourceKind.module_name;
        pub const nifs = .{
            e.ErlNifFunc{ .name = data_kind_name ++ ".memref_aligned", .arity = 1, .fptr = aligned_ptr, .flags = 0 },
            e.ErlNifFunc{ .name = data_kind_name ++ ".memref_opaque_ptr", .arity = 1, .fptr = self_opaque_ptr, .flags = 0 },
            e.ErlNifFunc{ .name = data_kind_name ++ ".memref_dump", .arity = 1, .fptr = memref_dump, .flags = 0 },
        } ++
            per_rank_resource_kinds[0].nifs ++
            per_rank_resource_kinds[1].nifs ++
            per_rank_resource_kinds[2].nifs ++
            per_rank_resource_kinds[3].nifs ++
            per_rank_resource_kinds[4].nifs;
        fn MemRefOfRank(comptime rank: u8) type {
            if (rank == 0) {
                return kinda.ResourceKind(UnrankMemRefDescriptor(ResourceKind), data_kind_name ++ ".MemRef." ++ @tagName(@intToEnum(MemRefRankType, 0)));
            } else {
                return kinda.ResourceKind(MemRefDescriptor(ResourceKind, rank), data_kind_name ++ ".MemRef." ++ @tagName(@intToEnum(MemRefRankType, rank)));
            }
        }
        const per_rank_resource_kinds = .{
            MemRefOfRank(0),
            MemRefOfRank(1),
            MemRefOfRank(2),
            MemRefOfRank(3),
            MemRefOfRank(4),
        };
        fn open(env: beam.env) void {
            comptime var i = 0;
            inline while (i < per_rank_resource_kinds.len) : (i += 1) {
                per_rank_resource_kinds[i].open_all(env);
            }
        }
    };
}

pub export const handwritten_nifs = .{
    e.ErlNifFunc{ .name = "beaver_raw_get_context_load_all_dialects", .arity = 0, .fptr = beaver_raw_get_context_load_all_dialects, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_registered_ops", .arity = 0, .fptr = beaver_raw_registered_ops, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_registered_ops_of_dialect", .arity = 1, .fptr = beaver_raw_registered_ops_of_dialect, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_registered_dialects", .arity = 0, .fptr = beaver_raw_registered_dialects, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_create_mlir_pass", .arity = 5, .fptr = beaver_raw_create_mlir_pass, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_pass_token_signal", .arity = 1, .fptr = PassToken.pass_token_signal, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_resource_c_string_to_term_charlist", .arity = 1, .fptr = beaver_raw_resource_c_string_to_term_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_attribute_to_charlist", .arity = 1, .fptr = beaver_raw_beaver_attribute_to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_type_to_charlist", .arity = 1, .fptr = beaver_raw_beaver_type_to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_operation_to_charlist", .arity = 1, .fptr = beaver_raw_beaver_operation_to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_get_resource_c_string", .arity = 1, .fptr = beaver_raw_get_resource_c_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_mlir_named_attribute_get", .arity = 2, .fptr = beaver_raw_mlir_named_attribute_get, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_read_opaque_ptr", .arity = 2, .fptr = beaver_raw_read_opaque_ptr, .flags = 0 },
} ++
    Complex.F32.nifs ++
    dataKindToMemrefKind(Complex.F32).nifs ++
    dataKindToMemrefKind(mlir_capi.U8).nifs ++
    dataKindToMemrefKind(mlir_capi.U16).nifs ++
    dataKindToMemrefKind(mlir_capi.U32).nifs ++
    dataKindToMemrefKind(mlir_capi.F32).nifs ++
    dataKindToMemrefKind(mlir_capi.F64).nifs ++
    dataKindToMemrefKind(mlir_capi.I8).nifs ++
    dataKindToMemrefKind(mlir_capi.I16).nifs ++
    dataKindToMemrefKind(mlir_capi.I32).nifs ++
    dataKindToMemrefKind(mlir_capi.I64).nifs;

pub export const num_nifs = mlir_capi.generated_nifs.len + handwritten_nifs.len;
pub export var nifs: [num_nifs]e.ErlNifFunc = handwritten_nifs ++ mlir_capi.generated_nifs;

const entry = e.ErlNifEntry{
    .major = 2,
    .minor = 16,
    .name = mlir_capi.root_module,
    .num_of_funcs = num_nifs,
    .funcs = &(nifs[0]),
    .load = nif_load,
    .reload = null, // currently unsupported
    .upgrade = null, // currently unsupported
    .unload = null, // currently unsupported
    .vm_variant = "beam.vanilla",
    .options = 1,
    .sizeof_ErlNifResourceTypeInit = @sizeOf(e.ErlNifResourceTypeInit),
    .min_erts = "erts-13.0",
};

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
    mlir_capi.open_generated_resource_types(env);
    comptime var i = 0;
    inline while (i < memref_kinds.len) : (i += 1) {
        memref_kinds[i].open(env);
    }
    Complex.F32.open_all(env);
    beam.open_resource_wrapped(env, PassToken);
    kinda.Internal.OpaqueStruct.open_all(env);
    return 0;
}

export fn nif_init() *const e.ErlNifEntry {
    return &entry;
}
