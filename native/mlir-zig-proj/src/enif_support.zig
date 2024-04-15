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

const enif_functions_otp26 = if (@hasDecl(e, "enif_get_string_length")) .{
    "enif_get_string_length",
    "enif_make_new_atom",
    "enif_make_new_atom_len",
    "enif_set_option",
} else .{};

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

const enif_function_names = .{
    "enif_alloc",
    "enif_alloc_binary",
    "enif_alloc_env",
    "enif_alloc_resource",
    "enif_binary_to_term",
    "enif_clear_env",
    "enif_compare",
    "enif_compare_monitors",
    "enif_compare_pids",
    "enif_cond_broadcast",
    "enif_cond_create",
    "enif_cond_destroy",
    "enif_cond_name",
    "enif_cond_signal",
    "enif_cond_wait",
    "enif_consume_timeslice",
    "enif_convert_time_unit",
    "enif_cpu_time",
    "enif_demonitor_process",
    "enif_dynamic_resource_call",
    "enif_equal_tids",
    "enif_fprintf",
    "enif_free",
    "enif_free_env",
    "enif_free_iovec",
    "enif_get_atom",
    "enif_get_atom_length",
    "enif_get_double",
    "enif_get_int",
    "enif_get_int64",
    "enif_get_list_cell",
    "enif_get_list_length",
    "enif_get_local_pid",
    "enif_get_local_port",
    "enif_get_long",
    "enif_get_map_size",
    "enif_get_map_value",
    "enif_get_resource",
    "enif_get_string",
    "enif_get_tuple",
    "enif_get_uint",
    "enif_get_uint64",
    "enif_get_ulong",
    "enif_getenv",
    "enif_has_pending_exception",
    "enif_hash",
    "enif_init_resource_type",
    "enif_inspect_binary",
    "enif_inspect_iolist_as_binary",
    "enif_inspect_iovec",
    "enif_ioq_create",
    "enif_ioq_deq",
    "enif_ioq_destroy",
    "enif_ioq_enq_binary",
    "enif_ioq_enqv",
    "enif_ioq_peek",
    "enif_ioq_peek_head",
    "enif_ioq_size",
    "enif_is_atom",
    "enif_is_binary",
    "enif_is_current_process_alive",
    "enif_is_empty_list",
    "enif_is_exception",
    "enif_is_fun",
    "enif_is_identical",
    "enif_is_list",
    "enif_is_map",
    "enif_is_number",
    "enif_is_pid",
    "enif_is_pid_undefined",
    "enif_is_port",
    "enif_is_port_alive",
    "enif_is_process_alive",
    "enif_is_ref",
    "enif_is_tuple",
    "enif_keep_resource",
    "enif_make_atom",
    "enif_make_atom_len",
    "enif_make_badarg",
    "enif_make_binary",
    "enif_make_copy",
    "enif_make_double",
    "enif_make_existing_atom",
    "enif_make_existing_atom_len",
    "enif_make_int",
    "enif_make_int64",
    "enif_make_list",
    "enif_make_list1",
    "enif_make_list2",
    "enif_make_list3",
    "enif_make_list4",
    "enif_make_list5",
    "enif_make_list6",
    "enif_make_list7",
    "enif_make_list8",
    "enif_make_list9",
    "enif_make_list_cell",
    "enif_make_list_from_array",
    "enif_make_long",
    "enif_make_map_from_arrays",
    "enif_make_map_put",
    "enif_make_map_remove",
    "enif_make_map_update",
    "enif_make_monitor_term",
    "enif_make_new_binary",
    "enif_make_new_map",
    // "enif_make_pid",
    "enif_make_ref",
    "enif_make_resource",
    "enif_make_resource_binary",
    "enif_make_reverse_list",
    "enif_make_string",
    "enif_make_string_len",
    "enif_make_sub_binary",
    "enif_make_tuple",
    "enif_make_tuple1",
    "enif_make_tuple2",
    "enif_make_tuple3",
    "enif_make_tuple4",
    "enif_make_tuple5",
    "enif_make_tuple6",
    "enif_make_tuple7",
    "enif_make_tuple8",
    "enif_make_tuple9",
    "enif_make_tuple_from_array",
    "enif_make_uint",
    "enif_make_uint64",
    "enif_make_ulong",
    "enif_make_unique_integer",
    "enif_map_iterator_create",
    "enif_map_iterator_destroy",
    "enif_map_iterator_get_pair",
    "enif_map_iterator_is_head",
    "enif_map_iterator_is_tail",
    "enif_map_iterator_next",
    "enif_map_iterator_prev",
    "enif_monitor_process",
    "enif_monotonic_time",
    "enif_mutex_create",
    "enif_mutex_destroy",
    "enif_mutex_lock",
    "enif_mutex_name",
    "enif_mutex_trylock",
    "enif_mutex_unlock",
    "enif_now_time",
    "enif_open_resource_type",
    "enif_open_resource_type_x",
    "enif_port_command",
    "enif_priv_data",
    "enif_raise_exception",
    "enif_realloc",
    "enif_realloc_binary",
    "enif_release_binary",
    "enif_release_resource",
    "enif_rwlock_create",
    "enif_rwlock_destroy",
    "enif_rwlock_name",
    "enif_rwlock_rlock",
    "enif_rwlock_runlock",
    "enif_rwlock_rwlock",
    "enif_rwlock_rwunlock",
    "enif_rwlock_tryrlock",
    "enif_rwlock_tryrwlock",
    "enif_schedule_nif",
    "enif_select",
    // "enif_select_read",
    // "enif_select_write",
    "enif_self",
    "enif_send",
    "enif_set_pid_undefined",
    "enif_sizeof_resource",
    "enif_snprintf",
    "enif_system_info",
    "enif_term_to_binary",
    "enif_term_type",
    "enif_thread_create",
    "enif_thread_exit",
    "enif_thread_join",
    "enif_thread_name",
    "enif_thread_opts_create",
    "enif_thread_opts_destroy",
    "enif_thread_self",
    "enif_thread_type",
    "enif_time_offset",
    "enif_tsd_get",
    "enif_tsd_key_create",
    "enif_tsd_key_destroy",
    "enif_tsd_set",
    "enif_vfprintf",
    "enif_vsnprintf",
    "enif_whereis_pid",
    "enif_whereis_port",
} ++ enif_functions_otp26 ++ beaver_runtime_functions;

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
