const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig").c;
const rt = @import("runtime.zig");
const kinda = @import("kinda");
const result = kinda.result;
const e = kinda.erl_nif;
const beam = kinda.beam;

const Invocation = struct {
    arg_terms: []beam.term = undefined,
    res_term: beam.term = undefined,
    packed_args: []?*anyopaque = undefined, // [arg0, arg1... result]
    fn init(self: *@This(), env: beam.env, list: beam.term) !void {
        const size = try beam.get_list_length(env, list);
        self.arg_terms = try beam.allocator.alloc(beam.term, size);
        errdefer beam.allocator.free(self.arg_terms);
        self.packed_args = try beam.allocator.alloc(?*anyopaque, size + 2);
        errdefer beam.allocator.free(self.packed_args);
        var movable_list = list;
        for (0..size) |idx| {
            self.arg_terms[idx] = try beam.get_head_and_iter(env, &movable_list);
            self.packed_args[idx + 1] = &self.arg_terms[idx];
        }
        self.packed_args[size + 1] = &self.res_term;
    }
    fn deinit(self: *@This()) void {
        beam.allocator.free(self.arg_terms);
        beam.allocator.free(self.packed_args);
    }
    fn invoke(self: *@This(), jit: mlir_capi.ExecutionEngine.T, name: beam.binary, env: beam.env) callconv(.c) mlir_capi.LogicalResult.T {
        // must set env here to make sure it stays alive during the call
        self.packed_args[0] = @ptrCast(@constCast(&env));
        return c.mlirExecutionEngineInvokePacked(jit, c.mlirStringRefCreate(name.data, name.size), &self.packed_args[0]);
    }
};

fn beaver_raw_jit_invoke_with_terms(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const Error = error{JITFunctionCallFailure};
    const jit: mlir_capi.ExecutionEngine.T = try mlir_capi.ExecutionEngine.resource.fetch(env, args[0]);
    const name: beam.binary = try beam.get_binary(env, args[1]);
    var invocation = Invocation{};
    try invocation.init(env, args[2]);
    defer invocation.deinit();
    if (c.beaverLogicalResultIsFailure(invocation.invoke(jit, name, env))) return Error.JITFunctionCallFailure;
    return invocation.res_term;
}

const enif_function_names = @import("enif_list.zig").functions ++ rt.exported;

fn register_jit_symbol(jit: mlir_capi.ExecutionEngine.T, comptime name: []const u8, comptime f: anytype) void {
    const name_str_ref = c.MlirStringRef{ .data = name.ptr, .length = name.len };
    c.mlirExecutionEngineRegisterSymbol(jit, name_str_ref, @ptrCast(@constCast(&f)));
}

// Route JIT allocation calls through the same C runtime std.c.free uses for
// deallocation. On Windows the JIT would otherwise resolve `malloc` from
// whichever loaded module exports it first, which may back a different heap
// than Zig's linked C runtime, crashing `free` on the returned pointer.
fn jit_malloc(size: usize) callconv(.c) ?*anyopaque {
    return std.c.malloc(size);
}
fn jit_free(ptr: ?*anyopaque) callconv(.c) void {
    std.c.free(ptr);
}
fn jit_realloc(ptr: ?*anyopaque, size: usize) callconv(.c) ?*anyopaque {
    return std.c.realloc(ptr, size);
}
fn jit_calloc(num: usize, size: usize) callconv(.c) ?*anyopaque {
    return std.c.calloc(num, size);
}

fn beaver_raw_jit_register_enif(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const jit = try mlir_capi.ExecutionEngine.resource.fetch(env, args[0]);
    inline for (enif_function_names) |name| {
        const f = if (@hasDecl(e, name)) @field(e, name) else @field(rt, name);
        register_jit_symbol(jit, "_mlir_ciface_" ++ name, f);
    }
    inline for (.{
        .{ "malloc", jit_malloc },
        .{ "free", jit_free },
        .{ "realloc", jit_realloc },
        .{ "calloc", jit_calloc },
    }) |symbol| {
        register_jit_symbol(jit, symbol[0], symbol[1]);
    }
    return beam.make_ok(env);
}

fn parse_mlir_type(env: beam.env, ctx: mlir_capi.Context.T, t: []const u8) !beam.term {
    return try mlir_capi.Type.resource.make(env, c.mlirTypeParseGet(ctx, c.MlirStringRef{
        .data = t.ptr,
        .length = t.len,
    }));
}

fn enif_mlir_type(env: beam.env, ctx: mlir_capi.Context.T, comptime t: type) !beam.term {
    if (t == rt.BinaryMemRefDescriptor) return parse_mlir_type(env, ctx, rt.BinaryMemRefType);
    if (t == beam.binary) return parse_mlir_type(env, ctx, rt.BinaryStructLLVMType);
    const mlir_type = switch (@typeInfo(t)) {
        .pointer, .optional => return parse_mlir_type(env, ctx, "!ptr.ptr<#ptr.generic_space>"),
        .int, .@"struct", .@"opaque" => c.mlirIntegerTypeGet(ctx, @bitSizeOf(t)),
        .float => switch (@bitSizeOf(t)) {
            32 => c.mlirF32TypeGet(ctx),
            64 => c.mlirF64TypeGet(ctx),
            else => @compileError("unsupported float type"),
        },
        .void => return beam.make_atom(env, "void"),
        else => @compileError("not supported type in enif signature: " ++ @typeName(t)),
    };
    return mlir_capi.Type.resource.make(env, mlir_type);
}

fn dump_type_info(env: beam.env, ctx: mlir_capi.Context.T, comptime t: type) !beam.term {
    var terms = [_]beam.term{ try enif_mlir_type(env, ctx, t), try beam.make(i64, env, @sizeOf(t)) };
    return beam.make_tuple(env, &terms);
}

fn beaver_raw_enif_signatures(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
    var signatures: [enif_function_names.len]beam.term = undefined;
    inline for (enif_function_names, 0..) |name, i| {
        const decl_name = "__decl__" ++ name;
        const f = if (@hasDecl(rt, decl_name)) @field(rt, decl_name) else (if (@hasDecl(e, name)) @field(e, name) else @field(rt, name));
        const info = @typeInfo(@TypeOf(f)).@"fn";
        var arguments: [info.params.len]beam.term = undefined;
        inline for (info.params, 0..) |param, index| {
            // translate-C exposes this one macro as a generic Zig function.
            const t = comptime param.type orelse if (std.mem.eql(u8, name, "enif_compare_pids")) [*c]u8 else @compileError("param type not found, function: " ++ name);
            arguments[index] = try dump_type_info(env, ctx, t);
        }
        const Return = comptime info.return_type orelse if (std.mem.eql(u8, name, "enif_compare_pids")) c_int else @compileError("return type not found, function: " ++ name);
        var returns: [if (Return == void) 0 else 1]beam.term = undefined;
        if (Return != void) returns[0] = try dump_type_info(env, ctx, Return);
        var fields = [_]beam.term{
            beam.make_atom(env, name),
            beam.make_term_list(env, &arguments),
            beam.make_term_list(env, &returns),
        };
        signatures[i] = beam.make_tuple(env, &fields);
    }
    return beam.make_term_list(env, &signatures);
}

fn beaver_raw_enif_functions(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    var names: [enif_function_names.len]beam.term = undefined;
    inline for (enif_function_names, 0..) |name, i| names[i] = beam.make_atom(env, name);
    return beam.make_term_list(env, &names);
}

fn beaver_raw_mlir_type_of_enif_obj(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const Error = error{MLIRTypeForEnifObjNotFound};
    const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
    const name = try beam.get_atom_slice(env, args[1]);
    inline for (.{ "term", "env", "binary", "pid" }) |obj| {
        if (@import("std").mem.eql(u8, name, obj)) {
            const t = @field(beam, obj);
            return try enif_mlir_type(env, ctx, t);
        }
    }
    return Error.MLIRTypeForEnifObjNotFound;
}

pub const nifs = .{
    result.nif_with_flags("beaver_raw_jit_invoke_with_terms", 3, beaver_raw_jit_invoke_with_terms, e.ERL_NIF_DIRTY_JOB_CPU_BOUND).entry,
    result.nif_with_flags("beaver_raw_jit_invoke_with_terms_cpu_bound", 3, beaver_raw_jit_invoke_with_terms, e.ERL_NIF_DIRTY_JOB_CPU_BOUND).entry,
    result.nif_with_flags("beaver_raw_jit_invoke_with_terms_io_bound", 3, beaver_raw_jit_invoke_with_terms, e.ERL_NIF_DIRTY_JOB_IO_BOUND).entry,
    result.nif("beaver_raw_jit_register_enif", 1, beaver_raw_jit_register_enif).entry,
    result.nif("beaver_raw_enif_signatures", 1, beaver_raw_enif_signatures).entry,
    result.nif("beaver_raw_enif_functions", 0, beaver_raw_enif_functions).entry,
    result.nif("beaver_raw_mlir_type_of_enif_obj", 2, beaver_raw_mlir_type_of_enif_obj).entry,
};
