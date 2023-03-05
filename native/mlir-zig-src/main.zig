const std = @import("std");
const io = std.io;
const stderr = io.getStdErr().writer();
const testing = std.testing;
const beam = @import("beam.zig");
const kinda = @import("kinda.zig");
const e = @import("erl_nif.zig");
const mlir_capi = @import("beaver.imp.zig");
pub const c = mlir_capi.c;

pub fn make_charlist_from_string_ref(environment: beam.env, val: mlir_capi.StringRef.T) beam.term {
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

fn get_context_load_all_dialects() mlir_capi.Context.T {
    const ctx = c.mlirContextCreate();
    var registry = c.mlirDialectRegistryCreate();
    c.mlirRegisterAllDialects(registry);
    c.mlirContextAppendDialectRegistry(ctx, registry);
    c.mlirDialectRegistryDestroy(registry);
    c.mlirContextLoadAllAvailableDialects(ctx);
    return ctx;
}

fn get_all_registered_ops2(env: beam.env, dialect: mlir_capi.StringRef.T) !beam.term {
    const ctx = get_context_load_all_dialects();
    defer c.mlirContextDestroy(ctx);
    var num_op: usize = 0;
    // TODO: refactor this dirty trick
    var names: [300]c.MlirRegisteredOperationName = undefined;
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
    var names: [300]mlir_capi.StringRef.T = undefined;
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
    var dialect: mlir_capi.StringRef.T = undefined;
    if (beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[0])) |value| {
        dialect = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for dialect, expected: mlir_capi.StringRef.T");
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
    var arg0: mlir_capi.Identifier.T = undefined;
    if (beam.fetch_resource(mlir_capi.Identifier.T, env, mlir_capi.Identifier.resource.t, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #0, expected: mlir_capi.Identifier.T");
    }
    var arg1: c.MlirAttribute = undefined;
    if (beam.fetch_resource(c.MlirAttribute, env, mlir_capi.Attribute.resource.t, args[1])) |value| {
        arg1 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.MlirAttribute");
    }

    var ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.NamedAttribute.resource.t, @sizeOf(mlir_capi.NamedAttribute.T));

    const RType = mlir_capi.NamedAttribute.T;
    var obj: *RType = undefined;

    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
        obj.* = mlir_capi.NamedAttribute.T{ .name = arg0, .attribute = arg1 };
    }
    return e.enif_make_resource(env, ptr);
}

const StringRefCollector = struct { env: beam.env, list: std.ArrayList(beam.term) };

fn collect_string_ref(string_ref: mlir_capi.StringRef.T, collector: ?*anyopaque) callconv(.C) void {
    var collector_ptr = @ptrCast(*StringRefCollector, @alignCast(@alignOf(*StringRefCollector), collector));
    collector_ptr.*.list.append(make_charlist_from_string_ref(collector_ptr.*.env, string_ref)) catch unreachable;
}

fn print_mlir(env: beam.env, element: anytype, printer: anytype) beam.term {
    var list = std.ArrayList(beam.term).init(beam.allocator);
    var collector = StringRefCollector{ .env = env, .list = list };
    defer list.deinit();
    if (element.ptr == null) {
        return beam.make_error_binary(env, "null pointer found: " ++ @typeName(@TypeOf(element)));
    }
    printer(element, collect_string_ref, &collector);
    return beam.make_term_list(env, collector.list.items);
}

fn Printer(comptime ResourceKind: type, comptime print_fn: anytype) type {
    return struct {
        fn to_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            var arg0: ResourceKind.T = ResourceKind.resource.fetch(env, args[0]) catch
                return beam.make_error_binary(env, "fail to fetch resource for argument #0, expected: " ++ @typeName(ResourceKind.T));
            return print_mlir(env, arg0, print_fn);
        }
    };
}

export fn beaver_raw_get_context_load_all_dialects(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    var ptr: ?*anyopaque = e.enif_alloc_resource(mlir_capi.Context.resource.t, @sizeOf(mlir_capi.Context.T));
    const RType = mlir_capi.Context.T;
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
    fn initialize(_: mlir_capi.Context.T, _: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        return mlir_capi.LogicalResult.T{ .value = 1 };
    }
    fn clone(userData: ?*anyopaque) callconv(.C) ?*anyopaque {
        const old = @ptrCast(*UserData, @alignCast(@alignOf(*UserData), userData));
        var new = beam.allocator.create(UserData) catch unreachable;
        new.* = old.*;
        return new;
    }
    fn run(op: mlir_capi.Operation.T, pass: c.MlirExternalPass, userData: ?*anyopaque) callconv(.C) void {
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
        tuple_slice[1] = beam.make_resource(env, op, mlir_capi.Operation.resource.t) catch {
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

const PtrOwner = extern struct {
    pub const Kind = kinda.ResourceKind(@This(), "Elixir.Beaver.Native.PtrOwner");
    ptr: mlir_capi.OpaquePtr.T,
    extern fn free(ptr: ?*anyopaque) void;
    pub fn destroy(_: beam.env, resource_ptr: ?*anyopaque) callconv(.C) void {
        const this_ptr = @ptrCast(*@This(), @alignCast(@alignOf(@This()), resource_ptr));
        print("destroy {}.\n", .{this_ptr});
        free(this_ptr.*.ptr);
    }
};

export fn beaver_raw_own_opaque_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var ptr: mlir_capi.OpaquePtr.T = mlir_capi.OpaquePtr.resource.fetch(env, args[0]) catch
        return beam.make_error_binary(env, "fail to fetch resource for ptr, expected: " ++ @typeName(mlir_capi.OpaquePtr.T));
    var owner: PtrOwner = .{ .ptr = ptr };
    return PtrOwner.Kind.resource.make(env, owner) catch return beam.make_error_binary(env, "fail to make owner");
}

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
    var name: mlir_capi.StringRef.T = undefined;
    if (beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[0])) |value| {
        name = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass name, expected: mlir_capi.StringRef.T");
    }
    var argument: mlir_capi.StringRef.T = undefined;
    if (beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[1])) |value| {
        argument = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass argument, expected: mlir_capi.StringRef.T");
    }
    var description: mlir_capi.StringRef.T = undefined;
    if (beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[2])) |value| {
        description = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass description, expected: mlir_capi.StringRef.T");
    }
    var op_name: mlir_capi.StringRef.T = undefined;
    if (beam.fetch_resource(mlir_capi.StringRef.T, env, mlir_capi.StringRef.resource.t, args[3])) |value| {
        op_name = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass op name, expected: mlir_capi.StringRef.T");
    }
    var handler: beam.pid = beam.get_pid(env, args[4]) catch return beam.make_error_binary(env, "expect the handler to be a pid");

    const typeIDAllocator = c.mlirTypeIDAllocatorCreate();
    defer c.mlirTypeIDAllocatorDestroy(typeIDAllocator);
    const passID = c.mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
    const nDependentDialects = 0;
    const dependentDialects = 0;
    var userData: *BeaverPass.UserData = beam.allocator.create(BeaverPass.UserData) catch return beam.make_error_binary(env, "fail to allocate for pass userdata");
    userData.*.handler = handler;
    const RType = mlir_capi.Pass.T;
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

fn MemRefDescriptorAccessor(comptime MemRefT: type) type {
    return struct {
        fn allocated_ptr(env: beam.env, term: beam.term) callconv(.C) beam.term {
            var descriptor: MemRefT = beam.fetch_resource(MemRefT, env, MemRefT.resource_type, term) catch
                return beam.make_error_binary(env, "fail to fetch resource for descriptor, expected: " ++ @typeName(MemRefT));
            var ret: mlir_capi.OpaquePtr.T = @ptrCast(mlir_capi.OpaquePtr.T, descriptor.allocated);
            return mlir_capi.OpaquePtr.resource.make(env, ret) catch return beam.make_error_binary(env, "fail to make allocated ptr");
        }
        fn aligned_ptr(env: beam.env, term: beam.term) callconv(.C) beam.term {
            var descriptor: MemRefT = beam.fetch_resource(MemRefT, env, MemRefT.resource_type, term) catch
                return beam.make_error_binary(env, "fail to fetch resource for descriptor, expected: " ++ @typeName(MemRefT));
            var ret: mlir_capi.OpaquePtr.T = @ptrCast(mlir_capi.OpaquePtr.T, descriptor.aligned);
            return mlir_capi.OpaquePtr.resource.make(env, ret) catch return beam.make_error_binary(env, "fail to make aligned ptr");
        }
        fn fetch_ptr_or_nil(env: beam.env, ptr_term: beam.term) !MemRefT.ElementResourceKind.Ptr.T {
            if (MemRefT.ElementResourceKind.Ptr.resource.fetch(env, ptr_term)) |value| {
                return value;
            } else |err| {
                if (try beam.is_nil(env, ptr_term)) {
                    return null;
                } else {
                    return err;
                }
            }
        }
        fn offset(env: beam.env, term: beam.term) callconv(.C) beam.term {
            var descriptor: MemRefT = beam.fetch_resource(MemRefT, env, MemRefT.resource_type, term) catch
                return beam.make_error_binary(env, "fail to fetch resource for descriptor, expected: " ++ @typeName(MemRefT));
            return beam.make_i64(env, descriptor.offset);
        }
    };
}

fn UnrankMemRefDescriptor(comptime ResourceKind: type) type {
    return extern struct {
        pub fn make(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            var allocated: ResourceKind.Ptr.T = MemRefDescriptorAccessor(@This()).fetch_ptr_or_nil(env, args[0]) catch
                return beam.make_error_binary(env, "fail to fetch allocated. expected: " ++ @typeName(ResourceKind.Ptr.T));
            var aligned: ResourceKind.Ptr.T = MemRefDescriptorAccessor(@This()).fetch_ptr_or_nil(env, args[1]) catch
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
        pub const module_name = ResourceKind.module_name ++ ".MemRef." ++ @tagName(@intToEnum(MemRefRankType, 0));
        const ElementResourceKind = ResourceKind;
        const T = ResourceKind.T;
        allocated: ?*T = null,
        aligned: ?*T = null,
        offset: i64 = undefined,
        pub var resource_type: beam.resource_type = undefined;
        fn allocated_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            return MemRefDescriptorAccessor(@This()).allocated_ptr(env, args[0]);
        }
        fn aligned_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            return MemRefDescriptorAccessor(@This()).aligned_ptr(env, args[0]);
        }
        fn get_offset(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            return MemRefDescriptorAccessor(@This()).offset(env, args[0]);
        }
        pub const nifs = .{ e.ErlNifFunc{ .name = module_name ++ ".allocated", .arity = 1, .fptr = allocated_ptr, .flags = 1 }, e.ErlNifFunc{ .name = module_name ++ ".aligned", .arity = 1, .fptr = aligned_ptr, .flags = 1 }, e.ErlNifFunc{ .name = module_name ++ ".offset", .arity = 1, .fptr = get_offset, .flags = 1 } };
    };
}

const BeaverDiagnostic = struct {
    pub fn printToStderr(str: mlir_capi.StringRef.T, _: ?*anyopaque) callconv(.C) void {
        stderr.print("{s}", .{str.data[0..str.length]}) catch return;
    }
    pub fn deleteUserData(_: ?*anyopaque) callconv(.C) void {}
    pub fn errorHandler(diagnostic: c.MlirDiagnostic, _: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        stderr.print("{s}", .{"[Beaver] [Diagnostic] ["}) catch return c.mlirLogicalResultFailure();
        const loc = c.mlirDiagnosticGetLocation(diagnostic);
        c.mlirLocationPrint(loc, printToStderr, null);
        stderr.print("{s}", .{"] "}) catch return c.mlirLogicalResultFailure();

        c.mlirDiagnosticPrint(diagnostic, printToStderr, null);
        stderr.print("{s}", .{"\n"}) catch return c.mlirLogicalResultFailure();

        const num_note = c.mlirDiagnosticGetNumNotes(diagnostic);
        var i: isize = 0;
        while (i < num_note) {
            const note_d = c.mlirDiagnosticGetNote(diagnostic, i);
            c.mlirDiagnosticPrint(note_d, printToStderr, null);
            i += 1;
        }
        stderr.print("{s}", .{"\n"}) catch return c.mlirLogicalResultFailure();
        return c.mlirLogicalResultSuccess();
    }
};

export fn beaver_raw_context_attach_diagnostic_handler(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: mlir_capi.Context.T = mlir_capi.Context.resource.fetch(env, args[0]) catch
        return beam.make_error_binary(env, "fail to fetch resource for argument #0, expected: " ++ @typeName(mlir_capi.Context.T));
    return mlir_capi.U64.resource.make(env, c.mlirContextAttachDiagnosticHandler(arg0, BeaverDiagnostic.errorHandler, null, BeaverDiagnostic.deleteUserData)) catch return beam.make_error_binary(env, "fail to make resource for: " ++ @typeName(mlir_capi.U64.T));
}

fn beaver_raw_parse_pass_pipeline(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
    var passManager: mlir_capi.MlirOpPassManager.T = mlir_capi.MlirOpPassManager.resource.fetch(env, args[0]) catch
        return beam.make_error_binary(env, "when calling C function mlirParsePassPipeline, fail to fetch resource for passManager, expected: " ++ @typeName(mlir_capi.MlirOpPassManager.T));
    var pipeline: mlir_capi.StringRef.T = mlir_capi.StringRef.resource.fetch(env, args[1]) catch
        return beam.make_error_binary(env, "when calling C function mlirParsePassPipeline, fail to fetch resource for pipeline, expected: " ++ @typeName(mlir_capi.StringRef.T));
    return mlir_capi.LogicalResult.resource.make(env, c.mlirOpPassManagerAddPipeline(passManager, pipeline, BeaverDiagnostic.printToStderr, null)) catch return beam.make_error_binary(env, "when calling C function mlirParsePassPipeline, fail to make resource for: " ++ @typeName(mlir_capi.LogicalResult.T));
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
            var allocated: ResourceKind.Ptr.T = MemRefDescriptorAccessor(@This()).fetch_ptr_or_nil(env, args[0]) catch
                return beam.make_error_binary(env, "fail to fetch allocated. expected: " ++ @typeName(ResourceKind.Ptr.T));
            var aligned: ResourceKind.Ptr.T = MemRefDescriptorAccessor(@This()).fetch_ptr_or_nil(env, args[1]) catch
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
        pub const ElementResourceKind = ResourceKind;
        pub const module_name = ResourceKind.module_name ++ ".MemRef." ++ @tagName(@intToEnum(MemRefRankType, N));
        pub var resource_type: beam.resource_type = undefined;
        fn allocated_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            return MemRefDescriptorAccessor(@This()).allocated_ptr(env, args[0]);
        }
        fn aligned_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            return MemRefDescriptorAccessor(@This()).aligned_ptr(env, args[0]);
        }
        fn get_offset(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            return MemRefDescriptorAccessor(@This()).offset(env, args[0]);
        }
        fn get_sizes(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            comptime var rank = N;
            var descriptor: @This() = beam.fetch_resource(@This(), env, @This().resource_type, args[0]) catch
                return beam.make_error_binary(env, "fail to fetch resource for descriptor, expected: " ++ @typeName(@This()));
            var ret: []beam.term = beam.allocator.alloc(beam.term, @intCast(usize, rank)) catch
                return beam.make_error_binary(env, "fail to allocate");
            defer beam.allocator.free(ret);
            var i: usize = 0;
            while (i < rank) : ({
                i += 1;
            }) {
                ret[@intCast(usize, i)] = beam.make_i64(env, descriptor.sizes[i]);
            }
            return beam.make_term_list(env, ret);
        }
        fn get_strides(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            comptime var rank = N;
            var descriptor: @This() = beam.fetch_resource(@This(), env, @This().resource_type, args[0]) catch
                return beam.make_error_binary(env, "fail to fetch resource for descriptor, expected: " ++ @typeName(@This()));
            var ret: []beam.term = beam.allocator.alloc(beam.term, @intCast(usize, rank)) catch
                return beam.make_error_binary(env, "fail to allocate");
            defer beam.allocator.free(ret);
            var i: usize = 0;
            while (i < rank) : ({
                i += 1;
            }) {
                ret[@intCast(usize, i)] = beam.make_i64(env, descriptor.strides[i]);
            }
            return beam.make_term_list(env, ret);
        }
        pub const nifs = .{ e.ErlNifFunc{ .name = module_name ++ ".allocated", .arity = 1, .fptr = allocated_ptr, .flags = 1 }, e.ErlNifFunc{ .name = module_name ++ ".aligned", .arity = 1, .fptr = aligned_ptr, .flags = 1 }, e.ErlNifFunc{ .name = module_name ++ ".offset", .arity = 1, .fptr = get_offset, .flags = 1 }, e.ErlNifFunc{ .name = module_name ++ ".sizes", .arity = 1, .fptr = get_sizes, .flags = 1 }, e.ErlNifFunc{ .name = module_name ++ ".strides", .arity = 1, .fptr = get_strides, .flags = 1 } };
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

fn dataTypeToResourceKind(comptime self: MemRefDataType) type {
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
        const data_kind_name = ResourceKind.module_name;
        pub const nifs =
            per_rank_resource_kinds[0].nifs ++
            per_rank_resource_kinds[1].nifs ++
            per_rank_resource_kinds[2].nifs ++
            per_rank_resource_kinds[3].nifs ++
            per_rank_resource_kinds[4].nifs ++
            per_rank_resource_kinds[5].nifs;
        fn MemRefOfRank(comptime rank: u8) type {
            if (rank == 0) {
                return kinda.ResourceKind2(UnrankMemRefDescriptor(ResourceKind));
            } else {
                return kinda.ResourceKind2(MemRefDescriptor(ResourceKind, rank));
            }
        }
        const per_rank_resource_kinds = .{
            MemRefOfRank(0),
            MemRefOfRank(1),
            MemRefOfRank(2),
            MemRefOfRank(3),
            MemRefOfRank(4),
            MemRefOfRank(5),
        };
        fn open(env: beam.env) void {
            comptime var i = 0;
            inline while (i < per_rank_resource_kinds.len) : (i += 1) {
                per_rank_resource_kinds[i].open_all(env);
            }
        }
    };
}

const handwritten_nifs = .{
    e.ErlNifFunc{ .name = "beaver_raw_get_context_load_all_dialects", .arity = 0, .fptr = beaver_raw_get_context_load_all_dialects, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_registered_ops", .arity = 0, .fptr = beaver_raw_registered_ops, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_registered_ops_of_dialect", .arity = 1, .fptr = beaver_raw_registered_ops_of_dialect, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_registered_dialects", .arity = 0, .fptr = beaver_raw_registered_dialects, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_create_mlir_pass", .arity = 5, .fptr = beaver_raw_create_mlir_pass, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_pass_token_signal", .arity = 1, .fptr = PassToken.pass_token_signal, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_context_attach_diagnostic_handler", .arity = 1, .fptr = beaver_raw_context_attach_diagnostic_handler, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_resource_c_string_to_term_charlist", .arity = 1, .fptr = beaver_raw_resource_c_string_to_term_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_attribute_to_charlist", .arity = 1, .fptr = Printer(mlir_capi.Attribute, c.mlirAttributePrint).to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_type_to_charlist", .arity = 1, .fptr = Printer(mlir_capi.Type, c.mlirTypePrint).to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_operation_to_charlist", .arity = 1, .fptr = Printer(mlir_capi.Operation, c.mlirOperationPrint).to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_value_to_charlist", .arity = 1, .fptr = Printer(mlir_capi.Value, c.mlirValuePrint).to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_pm_to_charlist", .arity = 1, .fptr = Printer(mlir_capi.MlirOpPassManager, c.mlirPrintPassPipeline).to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_affine_map_to_charlist", .arity = 1, .fptr = Printer(mlir_capi.AffineMap, c.mlirAffineMapPrint).to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_beaver_location_to_charlist", .arity = 1, .fptr = Printer(mlir_capi.Location, c.beaverLocationPrint).to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_get_resource_c_string", .arity = 1, .fptr = beaver_raw_get_resource_c_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_mlir_named_attribute_get", .arity = 2, .fptr = beaver_raw_mlir_named_attribute_get, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_own_opaque_ptr", .arity = 1, .fptr = beaver_raw_own_opaque_ptr, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_read_opaque_ptr", .arity = 2, .fptr = beaver_raw_read_opaque_ptr, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_parse_pass_pipeline", .arity = 2, .fptr = beaver_raw_parse_pass_pipeline, .flags = 0 },
} ++
    PtrOwner.Kind.nifs ++
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

const num_nifs = mlir_capi.generated_nifs.len + handwritten_nifs.len;
export var nifs: [num_nifs]e.ErlNifFunc = handwritten_nifs ++ mlir_capi.generated_nifs;

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
    kinda.open_internal_resource_types(env);
    mlir_capi.open_generated_resource_types(env);
    comptime var i = 0;
    inline while (i < memref_kinds.len) : (i += 1) {
        memref_kinds[i].open(env);
    }
    Complex.F32.open_all(env);
    beam.open_resource_wrapped(env, PassToken);
    kinda.Internal.OpaqueStruct.open_all(env);
    PtrOwner.Kind.open(env);
    return 0;
}

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

export fn nif_init() *const e.ErlNifEntry {
    return &entry;
}
