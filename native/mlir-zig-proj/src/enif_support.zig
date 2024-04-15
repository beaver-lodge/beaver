const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("runtime.zig");
const Invocation = struct {
    arg_terms: []beam.term = undefined,
    res_term: beam.term = undefined,
    packed_args: []?*anyopaque = undefined, // [arg0, arg1... result]
    pub fn init(self: *@This(), environment: beam.env, list: beam.term) !void {
        const size = try beam.get_list_length(environment, list);
        var head: beam.term = undefined;
        self.arg_terms = try beam.allocator.alloc(beam.term, size);
        self.packed_args = try beam.allocator.alloc(?*anyopaque, size + 2);
        var movable_list = list;
        for (0..size) |idx| {
            head = try beam.get_head_and_iter(environment, &movable_list);
            self.arg_terms[idx] = head;
            self.packed_args[idx + 1] = &self.arg_terms[idx];
        }
        self.packed_args[size + 1] = &self.res_term;
        errdefer beam.allocator.free(self.arg_terms);
        errdefer beam.allocator.free(self.packed_args);
    }
    pub fn deinit(self: *@This()) void {
        beam.allocator.free(self.arg_terms);
        beam.allocator.free(self.packed_args);
    }
    pub fn invoke(self: *@This(), environment: beam.env, jit: mlir_capi.ExecutionEngine.T, name: beam.binary) callconv(.C) mlir_capi.LogicalResult.T {
        self.packed_args[0] = @ptrCast(@constCast(&environment));
        return c.mlirExecutionEngineInvokePacked(jit, c.mlirStringRefCreate(name.data, name.size), &self.packed_args[0]);
    }
};

pub fn beaver_raw_jit_invoke_with_terms(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
    var jit: mlir_capi.ExecutionEngine.T = mlir_capi.ExecutionEngine.resource.fetch(env, args[0]) catch
        return beam.make_error_binary(env, "fail to fetch resource for ExecutionEngine, expected: " ++ @typeName(mlir_capi.ExecutionEngine.T));
    var name: beam.binary = beam.get_binary(env, args[1]) catch
        return beam.make_error_binary(env, "fail to get binary for jit func name");
    var invocation = Invocation{};
    invocation.init(env, args[2]) catch return beam.make_error_binary(env, "fail to init jit invocation");
    defer invocation.deinit();
    if (c.beaverLogicalResultIsFailure(invocation.invoke(env, jit, name))) {
        return beam.make_error_binary(env, "fail to call jit function");
    }
    return invocation.res_term;
}

const beaver_runtime_functions = .{
    "print_i32",
    "print_u32",
    "print_i64",
    "print_u64",
    "print_f32",
    "print_f64",
    "print_open",
    "print_close",
    "print_comma",
    "print_newline",
};

const enif_function_names = @import("enif_list.zig").functions ++ beaver_runtime_functions;

fn register_jit_symbol(jit: mlir_capi.ExecutionEngine.T, comptime name: []const u8, comptime f: anytype) void {
    const prefixed_name = "_mlir_ciface_" ++ name;
    const name_str_ref = c.MlirStringRef{
        .data = prefixed_name.ptr,
        .length = prefixed_name.len,
    };
    c.mlirExecutionEngineRegisterSymbol(jit, name_str_ref, @ptrCast(@constCast(&f)));
}

pub fn beaver_raw_jit_register_enif(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
    var jit: mlir_capi.ExecutionEngine.T = mlir_capi.ExecutionEngine.resource.fetch(env, args[0]) catch
        return beam.make_error_binary(env, "fail to fetch resource for ExecutionEngine, expected: " ++ @typeName(mlir_capi.ExecutionEngine.T));
    inline for (enif_function_names) |name| {
        register_jit_symbol(jit, name, @field(e, name));
    }
    return beam.make_ok(env);
}

fn llvm_ptr_type(env: beam.env, ctx: mlir_capi.Context.T) !beam.term {
    const llvm_ptr = "!llvm.ptr";
    return try mlir_capi.Type.resource.make(env, c.mlirTypeParseGet(ctx, c.MlirStringRef{
        .data = llvm_ptr.ptr,
        .length = llvm_ptr.len,
    }));
}

fn mlir_i_type_of_size(env: beam.env, ctx: mlir_capi.Context.T, comptime t: type) !beam.term {
    return mlir_capi.Type.resource.make(env, c.mlirIntegerTypeGet(ctx, @bitSizeOf(t)));
}

fn mlir_f_type_of_size(env: beam.env, ctx: mlir_capi.Context.T, comptime t: type) !beam.term {
    return mlir_capi.Type.resource.make(env, c.mlirFloatTypeGet(ctx, @bitSizeOf(t)));
}

fn enif_mlir_type(env: beam.env, ctx: mlir_capi.Context.T, comptime t: type) !beam.term {
    switch (@typeInfo(t)) {
        .Pointer => {
            return llvm_ptr_type(env, ctx);
        },
        .Opaque => {
            return try mlir_i_type_of_size(env, ctx, t);
        },
        .Struct => {
            return try mlir_i_type_of_size(env, ctx, t);
        },
        .Optional => {
            return llvm_ptr_type(env, ctx);
        },
        else => {
            const is_int = t == c_int or t == c_ulong or t == c_long or t == beam.env or t == usize or t == c_uint or t == i32 or t == u32 or t == i64 or t == u64;
            const is_float = t == f32 or t == f64;
            const is_struct = t == beam.resource_type or t == e.ErlNifCond;
            if (is_int or is_struct) {
                return try mlir_i_type_of_size(env, ctx, t);
            } else if (is_float) {
                return try mlir_i_type_of_size(env, ctx, t);
            } else if (t == void) {
                return beam.make_atom(env, "void");
            } else if (t == ?*anyopaque) {
                return llvm_ptr_type(env, ctx);
            } else {
                @compileError("not supported type in enif signature: " ++ @typeName(t));
            }
        },
    }
}

fn dump_type_info(env: beam.env, ctx: mlir_capi.Context.T, comptime t: type) !beam.term {
    var type_info_slice: []beam.term = try beam.allocator.alloc(beam.term, 2);
    type_info_slice[0] = try enif_mlir_type(env, ctx, t);
    type_info_slice[1] = try beam.make(i64, env, @sizeOf(t));
    defer beam.allocator.free(type_info_slice);
    return beam.make_tuple(env, type_info_slice);
}

pub fn beaver_raw_enif_signatures(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
    const ctx = mlir_capi.Context.resource.fetch(env, args[0]) catch
        return beam.make_error_binary(env, "fail to fetch resource for argument #0, expected: " ++ @typeName(mlir_capi.Context.T));
    var signatures: []beam.term = beam.allocator.alloc(beam.term, enif_function_names.len) catch
        return beam.make_error_binary(env, "fail to allocate");
    inline for (enif_function_names, 0..) |name, i| {
        const f = @field(e, name);
        const FTI = @typeInfo(@TypeOf(f)).Fn;
        var signature_slice: []beam.term = beam.allocator.alloc(beam.term, 3) catch
            return beam.make_error_binary(env, "fail to allocate");
        defer beam.allocator.free(signature_slice);
        var arg_type_slice: []beam.term = beam.allocator.alloc(beam.term, FTI.params.len) catch
            return beam.make_error_binary(env, "fail to allocate");
        defer beam.allocator.free(arg_type_slice);
        inline for (FTI.params, 0..) |p, arg_i| {
            if (p.type) |t| {
                arg_type_slice[arg_i] = dump_type_info(env, ctx, t) catch
                    return beam.make_error_binary(env, "fail to allocate");
            } else if (@TypeOf(f) == @TypeOf(e.enif_compare_pids)) {
                arg_type_slice[arg_i] = dump_type_info(env, ctx, [*c]u8) catch return beam.make_error_binary(env, "fail to dump type");
            } else {
                @compileError("param type not found, function: " ++ name);
            }
        }
        // {name, [arg_types...], [ret_type]}
        signature_slice[0] = beam.make_atom(env, name);
        signature_slice[1] = beam.make_term_list(env, arg_type_slice);
        var ret_size: usize = 1;
        if (FTI.return_type) |t| {
            if (t == void) {
                ret_size = 0;
            }
        }
        var ret_slice: []beam.term = beam.allocator.alloc(beam.term, ret_size) catch
            return beam.make_error_binary(env, "fail to allocate");
        defer beam.allocator.free(ret_slice);
        if (FTI.return_type) |t| {
            if (t != void) {
                ret_slice[0] = dump_type_info(env, ctx, t) catch return beam.make_error_binary(env, "fail to dump type");
            }
        } else if (f == e.enif_compare_pids) {
            ret_slice[0] = dump_type_info(env, ctx, c_int) catch return beam.make_error_binary(env, "fail to dump type");
        } else {
            @compileError("return type not found, function: " ++ name);
        }
        signature_slice[2] = beam.make_term_list(env, ret_slice);
        signatures[i] = beam.make_tuple(env, signature_slice);
    }
    return beam.make_term_list(env, signatures);
}

pub fn beaver_raw_enif_functions(env: beam.env, _: c_int, _: [*c]const beam.term) callconv(.C) beam.term {
    var names: []beam.term = beam.allocator.alloc(beam.term, enif_function_names.len) catch
        return beam.make_error_binary(env, "fail to allocate");
    inline for (enif_function_names, 0..) |name, i| {
        names[i] = beam.make_atom(env, name);
    }
    return beam.make_term_list(env, names);
}

pub fn beaver_raw_mlir_type_of_enif_obj(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
    const ctx = mlir_capi.Context.resource.fetch(env, args[0]) catch
        return beam.make_error_binary(env, "fail to fetch resource for argument #0, expected: " ++ @typeName(mlir_capi.Context.T));
    var name = beam.get_atom_slice(env, args[1]) catch
        return beam.make_error_binary(env, "fail to get name");
    inline for (.{ "term", "env" }) |obj| {
        if (@import("std").mem.eql(u8, name, obj)) {
            const t = @field(beam, obj);
            return enif_mlir_type(env, ctx, t) catch return beam.make_error_binary(env, "fail to get mlir type");
        }
    }
    return beam.make_error_binary(env, "mlir type not found for enif obj");
}
