// Windows NIF imports share the callback table supplied by the emulator.
// Derive each forwarding signature from OTP instead of duplicating its ABI.
const e = @import("kinda").erl_nif;
export var WinDynNifCallbacks: e.TWinDynNifCallbacks = .{};
fn Forward(comptime name: []const u8) type {
    const Pointer = @typeInfo(@FieldType(e.TWinDynNifCallbacks, name)).optional.child;
    const signature = @typeInfo(@typeInfo(Pointer).pointer.child).@"fn";
    const R = signature.return_type.?;
    const P = struct {
        fn at(comptime index: usize) type {
            return signature.params[index].type.?;
        }
    };
    return switch (signature.params.len) {
        0 => struct {
            fn call() callconv(.c) R {
                return @field(e.WinDynNifCallbacks, name).?();
            }
        },
        1 => struct {
            fn call(a0: P.at(0)) callconv(.c) R {
                return @field(e.WinDynNifCallbacks, name).?(a0);
            }
        },
        2 => struct {
            fn call(a0: P.at(0), a1: P.at(1)) callconv(.c) R {
                return @field(e.WinDynNifCallbacks, name).?(a0, a1);
            }
        },
        3 => struct {
            fn call(a0: P.at(0), a1: P.at(1), a2: P.at(2)) callconv(.c) R {
                return @field(e.WinDynNifCallbacks, name).?(a0, a1, a2);
            }
        },
        4 => struct {
            fn call(a0: P.at(0), a1: P.at(1), a2: P.at(2), a3: P.at(3)) callconv(.c) R {
                return @field(e.WinDynNifCallbacks, name).?(a0, a1, a2, a3);
            }
        },
        5 => struct {
            fn call(a0: P.at(0), a1: P.at(1), a2: P.at(2), a3: P.at(3), a4: P.at(4)) callconv(.c) R {
                return @field(e.WinDynNifCallbacks, name).?(a0, a1, a2, a3, a4);
            }
        },
        6 => struct {
            fn call(a0: P.at(0), a1: P.at(1), a2: P.at(2), a3: P.at(3), a4: P.at(4), a5: P.at(5)) callconv(.c) R {
                return @field(e.WinDynNifCallbacks, name).?(a0, a1, a2, a3, a4, a5);
            }
        },
        7 => struct {
            fn call(a0: P.at(0), a1: P.at(1), a2: P.at(2), a3: P.at(3), a4: P.at(4), a5: P.at(5), a6: P.at(6)) callconv(.c) R {
                return @field(e.WinDynNifCallbacks, name).?(a0, a1, a2, a3, a4, a5, a6);
            }
        },
        else => @compileError("unsupported OTP callback arity: " ++ name),
    };
}
comptime {
    @setEvalBranchQuota(10000);
    for (@typeInfo(e.TWinDynNifCallbacks).@"struct".fields) |field| {
        const Pointer = @typeInfo(field.type).optional.child;
        const info = @typeInfo(@typeInfo(Pointer).pointer.child);
        if (info != .@"fn") continue;
        const signature = info.@"fn";
        // Variadic entrypoints retain their explicit fixed-arity adapters below.
        if (!signature.is_var_args) @export(&Forward(field.name).call, .{ .name = field.name });
    }
}
comptime {
    if (!@hasField(e.TWinDynNifCallbacks, "enif_get_int64")) {
        @export(&enif_get_int64, .{ .name = "enif_get_int64" });
        @export(&enif_get_uint64, .{ .name = "enif_get_uint64" });
        @export(&enif_make_int64, .{ .name = "enif_make_int64" });
        @export(&enif_make_uint64, .{ .name = "enif_make_uint64" });
    }
}
export fn enif_make_tuple(env: ?*e.ErlNifEnv, cnt: c_uint, t1: e.ERL_NIF_TERM, t2: e.ERL_NIF_TERM) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_tuple.?(env, cnt, t1, t2);
}
export fn enif_make_list(env: ?*e.ErlNifEnv, cnt: c_uint) e.ERL_NIF_TERM {
    return e.WinDynNifCallbacks.enif_make_list.?(env, cnt);
}
export fn enif_fprintf(filep: ?*e.FILE, format: [*c]const u8) c_int {
    return e.WinDynNifCallbacks.enif_fprintf.?(filep, format);
}
export fn enif_snprintf(buffer: [*c]u8, size: usize, format: [*c]const u8) c_int {
    return e.WinDynNifCallbacks.enif_snprintf.?(buffer, size, format);
}
export fn enif_set_option(env: ?*e.ErlNifEnv, opt: e.ErlNifOption) c_int {
    return e.WinDynNifCallbacks.enif_set_option.?(env, opt);
}

// The MSVC emulator keeps 64-bit signedness distinct from c_long, so these
// are real callbacks there. On LP64 ABIs they alias the long variants and
// translate-c emits aliases instead, so the callbacks table has no fields.
fn enif_get_int64(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, ip: [*c]e.ErlNifSInt64) callconv(.c) c_int {
    return if (@hasField(e.TWinDynNifCallbacks, "enif_get_int64"))
        e.WinDynNifCallbacks.enif_get_int64.?(env, term, ip)
    else
        e.enif_get_long(env, term, @ptrCast(ip));
}
fn enif_get_uint64(env: ?*e.ErlNifEnv, term: e.ERL_NIF_TERM, ip: [*c]e.ErlNifUInt64) callconv(.c) c_int {
    return if (@hasField(e.TWinDynNifCallbacks, "enif_get_uint64"))
        e.WinDynNifCallbacks.enif_get_uint64.?(env, term, ip)
    else
        e.enif_get_ulong(env, term, @ptrCast(ip));
}
fn enif_make_int64(env: ?*e.ErlNifEnv, value: e.ErlNifSInt64) callconv(.c) e.ERL_NIF_TERM {
    return if (@hasField(e.TWinDynNifCallbacks, "enif_make_int64"))
        e.WinDynNifCallbacks.enif_make_int64.?(env, value)
    else
        e.enif_make_long(env, @intCast(value));
}
fn enif_make_uint64(env: ?*e.ErlNifEnv, value: e.ErlNifUInt64) callconv(.c) e.ERL_NIF_TERM {
    return if (@hasField(e.TWinDynNifCallbacks, "enif_make_uint64"))
        e.WinDynNifCallbacks.enif_make_uint64.?(env, value)
    else
        e.enif_make_ulong(env, @intCast(value));
}
