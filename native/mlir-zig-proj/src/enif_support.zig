const beam = @import("beam");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const result = @import("kinda").result;
const e = @import("runtime.zig");

fn Invocation(
    comptime ArgCount: u8,
) type {
    return struct {
        arg_terms: [ArgCount]beam.term = undefined,
        res_term: beam.term = undefined,
        packed_args: [ArgCount]?*anyopaque = undefined, // [arg0, arg1... result]
        fn init(self: *@This(), environment: beam.env, list: beam.term) !void {
            self.packed_args[0] = @ptrCast(@constCast(&environment));
            const size = try beam.get_list_length(environment, list);
            var head: beam.term = undefined;
            var movable_list = list;
            for (0..size) |idx| {
                head = try beam.get_head_and_iter(environment, &movable_list);
                self.arg_terms[idx] = head;
                self.packed_args[idx + 1] = &self.arg_terms[idx];
            }
            self.packed_args[size + 1] = &self.res_term;
        }
        fn invoke(self: *@This(), jit: mlir_capi.ExecutionEngine.T, name: beam.binary) callconv(.C) mlir_capi.LogicalResult.T {
            return c.mlirExecutionEngineInvokePacked(jit, c.mlirStringRefCreate(name.data, name.size), &self.packed_args[0]);
        }
    };
}

fn beaver_raw_jit_invoke_with_terms(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const Error = error{JITFunctionCallFailure};
    const jit: mlir_capi.ExecutionEngine.T = try mlir_capi.ExecutionEngine.resource.fetch(env, args[0]);
    const name: beam.binary = try beam.get_binary(env, args[1]);
    var invocation = Invocation(16){};
    try invocation.init(env, args[2]);
    if (c.beaverLogicalResultIsFailure(invocation.invoke(jit, name))) {
        return Error.JITFunctionCallFailure;
    }
    return invocation.res_term;
}

const enif_function_names = @import("enif_list.zig").functions ++ e.exported;

fn register_jit_symbol(jit: mlir_capi.ExecutionEngine.T, comptime name: []const u8, comptime f: anytype) void {
    const prefixed_name = "_mlir_ciface_" ++ name;
    const name_str_ref = c.MlirStringRef{
        .data = prefixed_name.ptr,
        .length = prefixed_name.len,
    };
    c.mlirExecutionEngineRegisterSymbol(jit, name_str_ref, @ptrCast(@constCast(&f)));
}

fn beaver_raw_jit_register_enif(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const jit = try mlir_capi.ExecutionEngine.resource.fetch(env, args[0]);
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

fn beaver_raw_enif_signatures(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
    var signatures: []beam.term = try beam.allocator.alloc(beam.term, enif_function_names.len);
    inline for (enif_function_names, 0..) |name, i| {
        const f = @field(e, name);
        const FTI = @typeInfo(@TypeOf(f)).Fn;
        var signature_slice: []beam.term = try beam.allocator.alloc(beam.term, 3);
        defer beam.allocator.free(signature_slice);
        var arg_type_slice: []beam.term = try beam.allocator.alloc(beam.term, FTI.params.len);
        defer beam.allocator.free(arg_type_slice);
        inline for (FTI.params, 0..) |p, arg_i| {
            if (p.type) |t| {
                arg_type_slice[arg_i] = try dump_type_info(env, ctx, t);
            } else if (@TypeOf(f) == @TypeOf(e.enif_compare_pids)) {
                arg_type_slice[arg_i] = try dump_type_info(env, ctx, [*c]u8);
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
        var ret_slice: []beam.term = try beam.allocator.alloc(beam.term, ret_size);
        defer beam.allocator.free(ret_slice);
        if (FTI.return_type) |t| {
            if (t != void) {
                ret_slice[0] = try dump_type_info(env, ctx, t);
            }
        } else if (f == e.enif_compare_pids) {
            ret_slice[0] = try dump_type_info(env, ctx, c_int);
        } else {
            @compileError("return type not found, function: " ++ name);
        }
        signature_slice[2] = beam.make_term_list(env, ret_slice);
        signatures[i] = beam.make_tuple(env, signature_slice);
    }
    return beam.make_term_list(env, signatures);
}

fn beaver_raw_enif_functions(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    var names: []beam.term = try beam.allocator.alloc(beam.term, enif_function_names.len);
    inline for (enif_function_names, 0..) |name, i| {
        names[i] = beam.make_atom(env, name);
    }
    return beam.make_term_list(env, names);
}

fn beaver_raw_mlir_type_of_enif_obj(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const Error = error{MLIRTypeForEnifObjNotFound};
    const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
    const name = try beam.get_atom_slice(env, args[1]);
    inline for (.{ "term", "env" }) |obj| {
        if (@import("std").mem.eql(u8, name, obj)) {
            const t = @field(beam, obj);
            return try enif_mlir_type(env, ctx, t);
        }
    }
    return Error.MLIRTypeForEnifObjNotFound;
}

pub const nifs = .{
    result.nif("beaver_raw_jit_invoke_with_terms", 3, beaver_raw_jit_invoke_with_terms).entry,
    result.nif("beaver_raw_jit_register_enif", 1, beaver_raw_jit_register_enif).entry,
    result.nif("beaver_raw_enif_signatures", 1, beaver_raw_enif_signatures).entry,
    result.nif("beaver_raw_enif_functions", 0, beaver_raw_enif_functions).entry,
    result.nif("beaver_raw_mlir_type_of_enif_obj", 2, beaver_raw_mlir_type_of_enif_obj).entry,
};
