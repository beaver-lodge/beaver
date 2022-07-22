const std = @import("std");
const testing = std.testing;

export fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try testing.expect(add(3, 7) == 10);
}
const beam = @import("beam.zig");
const e = @import("erl_nif.zig");
const fizz = @import("mlir.imp.zig");
pub const c = fizz.c;

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
    var num_op: usize = 0;
    // TODO: refactor this dirty trick
    var names: [300]c.struct_MlirRegisteredOperationName = undefined;
    c.beaverRegisteredOperationsOfDialect(ctx, dialect, &names, &num_op);
    if (num_op == 0) {
        return beam.make_error_binary(env, "no ops found for dialect");
    }
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(usize, num_op));
    defer beam.allocator.free(ret);
    var i: usize = 0;
    while (i < num_op) : ({
        i += 1;
    }) {
        const registered_op_name = names[i];
        const op_name = c.beaverRegisteredOperationNameGetOpName(registered_op_name);
        ret[@intCast(usize, i)] = beam.make_cstring_charlist(env, op_name.data);
    }
    return beam.make_term_list(env, ret);
}

export fn registered_ops(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    return get_all_registered_ops(env) catch beam.make_error_binary(env, "launching nif");
}

export fn registered_ops_of_dialect(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var dialect: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, fizz.resource_type_c_struct_MlirStringRef, args[0])) |value| {
        dialect = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for dialect, expected: c.struct_MlirStringRef");
    }
    return get_all_registered_ops2(env, dialect) catch beam.make_error_binary(env, "launching nif");
}

export fn resource_cstring_to_term_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    const T = [*c]const u8;
    var arg0: T = undefined;
    if (beam.fetch_resource(T, env, fizz.resource_type__cptr_const_u8, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource");
    }
    return beam.make_cstring_charlist(env, arg0);
}

export fn resource_bool_to_term(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: bool = undefined;
    if (beam.fetch_resource(bool, env, fizz.resource_type_bool, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource");
    }
    return beam.make_bool(env, arg0);
}

export fn get_resource_bool(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var ptr: ?*anyopaque = e.enif_alloc_resource(fizz.resource_type_bool, @sizeOf(bool));
    var obj: *bool = undefined;
    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(*bool, @alignCast(@alignOf(*bool), ptr));
    }
    if (beam.get(bool, env, args[0])) |value| {
        obj.* = value;
        return e.enif_make_resource(env, ptr);
    } else |_| {
        return beam.make_error_binary(env, "launching nif");
    }
}

// create a C string resource by copying given binary
const mem = @import("std").mem;
// memory layout {address, real_binary, null}
export fn get_resource_c_string(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    const RType = [*c]u8;
    var bin: beam.binary = undefined;
    if (0 == e.enif_inspect_binary(env, args[0], &bin)) {
        return beam.make_error_binary(env, "not a binary");
    }
    var ptr: ?*anyopaque = e.enif_alloc_resource(fizz.resource_type__cptr_const_u8, @sizeOf(RType) + bin.size + 1);
    var obj: *RType = undefined;
    var real_binary: RType = undefined;
    if (ptr == null) {
        unreachable();
    } else {
        obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
        real_binary = @ptrCast(RType, ptr);
        real_binary += @alignOf(RType);
        real_binary[bin.size] = 0;
        obj.* = real_binary;
    }
    mem.copy(u8, real_binary[0..bin.size], bin.data[0..bin.size]);
    return e.enif_make_resource(env, ptr);
}

export fn beaver_nif_NamedAttributeGet(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: c.struct_MlirIdentifier = undefined;
    if (beam.fetch_resource(c.struct_MlirIdentifier, env, fizz.resource_type_c_struct_MlirIdentifier, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.struct_MlirIdentifier");
    }
    var arg1: c.struct_MlirAttribute = undefined;
    if (beam.fetch_resource(c.struct_MlirAttribute, env, fizz.resource_type_c_struct_MlirAttribute, args[1])) |value| {
        arg1 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #2, expected: c.struct_MlirAttribute");
    }

    var ptr: ?*anyopaque = e.enif_alloc_resource(fizz.resource_type_c_struct_MlirNamedAttribute, @sizeOf(c.struct_MlirNamedAttribute));

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

export fn beaver_attribute_to_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: c.struct_MlirAttribute = undefined;
    if (beam.fetch_resource(c.struct_MlirAttribute, env, fizz.resource_type_c_struct_MlirAttribute, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.struct_MlirAttribute");
    }
    return print_mlir(env, arg0, c.mlirAttributePrint);
}

export fn beaver_type_to_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: c.struct_MlirType = undefined;
    if (beam.fetch_resource(c.struct_MlirType, env, fizz.resource_type_c_struct_MlirType, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.struct_MlirType");
    }
    return print_mlir(env, arg0, c.mlirTypePrint);
}

export fn beaver_operation_to_charlist(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var arg0: c.struct_MlirOperation = undefined;
    if (beam.fetch_resource(c.struct_MlirOperation, env, fizz.resource_type_c_struct_MlirOperation, args[0])) |value| {
        arg0 = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for argument #1, expected: c.struct_MlirOperation");
    }
    return print_mlir(env, arg0, c.mlirOperationPrint);
}

export fn beaver_get_context_load_all_dialects(env: beam.env, _: c_int, _: [*c]const beam.term) beam.term {
    var ptr: ?*anyopaque = e.enif_alloc_resource(fizz.resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));
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
    lock: *std.Thread.Mutex,
    cond: *std.Thread.Condition,
    pub var resource_type: beam.resource_type = undefined;
    pub const resource_name = "Beaver" ++ @typeName(@This());
};

const print = @import("std").debug.print;
const BeaverPass = struct {
    const UserData = struct { handler: beam.pid };
    export fn construct(_: ?*anyopaque) callconv(.C) void {}

    export fn destruct(userData: ?*anyopaque) callconv(.C) void {
        const ptr = @ptrCast(*UserData, @alignCast(@alignOf(UserData), userData));
        beam.allocator.destroy(ptr);
    }
    export fn initialize(_: c.struct_MlirContext, _: ?*anyopaque) callconv(.C) c.struct_MlirLogicalResult {
        return c.struct_MlirLogicalResult{ .value = 1 };
    }
    export fn clone(userData: ?*anyopaque) callconv(.C) ?*anyopaque {
        const old = @ptrCast(*UserData, @alignCast(@alignOf(*UserData), userData));
        var new = beam.allocator.create(UserData) catch unreachable;
        new.* = old.*;
        return new;
    }
    export fn run(op: c.struct_MlirOperation, pass: c.struct_MlirExternalPass, userData: ?*anyopaque) callconv(.C) void {
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
        var tuple_slice: []beam.term = beam.allocator.alloc(beam.term, 3) catch unreachable;
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_atom(env, "run");
        tuple_slice[1] = beam.make_resource(env, op, fizz.resource_type_c_struct_MlirOperation) catch {
            print("fail to make res: {}\n", .{@TypeOf(op)});
            unreachable;
        };
        var mutex: std.Thread.Mutex = .{};
        var cond: std.Thread.Condition = .{};
        var token = PassToken{ .lock = &mutex, .cond = &cond };
        tuple_slice[2] = beam.make_resource_wrapped(env, token) catch {
            print("fail to make token: {}\n", .{@TypeOf(token)});
            unreachable;
        };
        if (!beam.send(env, handler, beam.make_tuple(env, tuple_slice))) {
            print("fail to send message to pass handler.\n", .{});
            c.mlirExternalPassSignalFailure(pass);
        }
    }
};

export fn beaver_raw_create_mlir_pass(env: beam.env, _: c_int, args: [*c]const beam.term) beam.term {
    var name: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, fizz.resource_type_c_struct_MlirStringRef, args[0])) |value| {
        name = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass name, expected: c.struct_MlirStringRef");
    }
    var argument: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, fizz.resource_type_c_struct_MlirStringRef, args[1])) |value| {
        argument = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass argument, expected: c.struct_MlirStringRef");
    }
    var description: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, fizz.resource_type_c_struct_MlirStringRef, args[2])) |value| {
        description = value;
    } else |_| {
        return beam.make_error_binary(env, "fail to fetch resource for pass description, expected: c.struct_MlirStringRef");
    }
    var op_name: c.struct_MlirStringRef = undefined;
    if (beam.fetch_resource(c.struct_MlirStringRef, env, fizz.resource_type_c_struct_MlirStringRef, args[3])) |value| {
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
    var ptr: ?*anyopaque = e.enif_alloc_resource(fizz.resource_type_c_struct_MlirPass, @sizeOf(RType));
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

pub export const handwritten_nifs = ([_]e.ErlNifFunc{
    e.ErlNifFunc{ .name = "beaver_get_context_load_all_dialects", .arity = 0, .fptr = beaver_get_context_load_all_dialects, .flags = 1 },
    e.ErlNifFunc{ .name = "registered_ops", .arity = 0, .fptr = registered_ops, .flags = 1 },
    e.ErlNifFunc{ .name = "registered_ops_of_dialect", .arity = 1, .fptr = registered_ops_of_dialect, .flags = 1 },
    e.ErlNifFunc{ .name = "beaver_raw_create_mlir_pass", .arity = 5, .fptr = beaver_raw_create_mlir_pass, .flags = 0 },
    e.ErlNifFunc{ .name = "resource_cstring_to_term_charlist", .arity = 1, .fptr = resource_cstring_to_term_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_attribute_to_charlist", .arity = 1, .fptr = beaver_attribute_to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_type_to_charlist", .arity = 1, .fptr = beaver_type_to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_operation_to_charlist", .arity = 1, .fptr = beaver_operation_to_charlist, .flags = 0 },
    e.ErlNifFunc{ .name = "resource_bool_to_term", .arity = 1, .fptr = resource_bool_to_term, .flags = 0 },
    e.ErlNifFunc{ .name = "get_resource_bool", .arity = 1, .fptr = get_resource_bool, .flags = 0 },
    e.ErlNifFunc{ .name = "get_resource_c_string", .arity = 1, .fptr = get_resource_c_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_nif_MlirNamedAttributeGet", .arity = 2, .fptr = beaver_nif_NamedAttributeGet, .flags = 0 },
});

pub export const num_nifs = fizz.generated_nifs.len + handwritten_nifs.len;
pub export var nifs: [num_nifs]e.ErlNifFunc = undefined;

const entry = e.ErlNifEntry{
    .major = 2,
    .minor = 16,
    .name = "Elixir.Beaver.MLIR.CAPI",
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
    fizz.open_generated_resource_types(env);
    beam.open_resource_wrapped(env, PassToken);
    return 0;
}

export fn nif_init() *const e.ErlNifEntry {
    var i: usize = 0;
    while (i < fizz.generated_nifs.len) : ({
        i += 1;
    }) {
        nifs[i] = fizz.generated_nifs[i];
    }
    var j: usize = 0;
    while (j < handwritten_nifs.len) : ({
        j += 1;
    }) {
        nifs[fizz.generated_nifs.len + j] = handwritten_nifs[j];
    }
    return &entry;
}
