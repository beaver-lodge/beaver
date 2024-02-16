const std = @import("std");
const beam = @import("beam");
const e = @import("erl_nif");
pub fn KindaLibrary(comptime Kinds: anytype, comptime NIFs: anytype) type {
    return struct {
        fn getKind(comptime t: type) type {
            switch (@typeInfo(t)) {
                .Pointer => {
                    for (Kinds) |kind| {
                        if (t == kind.Ptr.T) {
                            return kind.Ptr;
                        }
                        if (t == kind.Array.T) {
                            return kind.Array;
                        }
                    }
                },
                else => {
                    for (Kinds) |kind| {
                        if (t == kind.T) {
                            return kind;
                        }
                    }
                },
            }
            @compileError("resouce kind not found " ++ @typeName(t));
        }

        fn KindaNIF(comptime cfunction: anytype, comptime nif_name: [*c]const u8, comptime flags: c_uint) type {
            return struct {
                const FT = @typeInfo(@TypeOf(cfunction)).Fn;
                inline fn VariadicArgs() type {
                    const P = FT.params;
                    switch (P.len) {
                        0 => return struct {},
                        1 => return struct { P[0].type.? },
                        2 => return struct { P[0].type.?, P[1].type.? },
                        3 => return struct { P[0].type.?, P[1].type.?, P[2].type.? },
                        4 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.? },
                        5 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.? },
                        6 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.? },
                        7 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.? },
                        8 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.? },
                        9 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.? },
                        10 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.?, P[9].type.? },
                        11 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.?, P[9].type.?, P[10].type.? },
                        12 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.?, P[9].type.?, P[10].type.?, P[11].type.? },
                        13 => return struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.?, P[9].type.?, P[10].type.?, P[11].type.?, P[12].type.? },
                        else => {
                            var buffer: [20]u8 = undefined;
                            const s = std.fmt.bufPrint(&buffer, "{}", .{P.len}) catch unreachable;
                            @compileError("too many args " ++ s ++ ", fn: " ++ @typeName(@TypeOf(cfunction)));
                        },
                    }
                }
                inline fn variadic_call(args: anytype) FT.return_type.? {
                    const f = cfunction;
                    switch (FT.params.len) {
                        0 => return f(),
                        1 => return f(args[0]),
                        2 => return f(args[0], args[1]),
                        3 => return f(args[0], args[1], args[2]),
                        4 => return f(args[0], args[1], args[2], args[3]),
                        5 => return f(args[0], args[1], args[2], args[3], args[4]),
                        6 => return f(args[0], args[1], args[2], args[3], args[4], args[5]),
                        7 => return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6]),
                        8 => return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
                        9 => return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]),
                        10 => return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]),
                        11 => return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]),
                        12 => return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]),
                        13 => return f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]),
                        else => @compileError("too many args"),
                    }
                }
                fn nif(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
                    var c_args: VariadicArgs() = undefined;
                    inline for (0..FT.params.len) |i| {
                        const ArgKind = getKind(FT.params[i].type.?);
                        c_args[i] = ArgKind.resource.fetch(env, args[i]) catch
                            return beam.make_error_binary(env, "fail to fetch resource, expected: " ++ @typeName(ArgKind.T));
                    }
                    const rt = FT.return_type.?;
                    if (rt == void) {
                        variadic_call(c_args);
                        return beam.make_ok(env);
                    } else {
                        const RetKind = getKind(rt);
                        return RetKind.resource.make(env, variadic_call(c_args)) catch return beam.make_error_binary(env, "fail to make resource, type: " ++ @typeName(RetKind.T));
                    }
                }
                const entry = e.ErlNifFunc{ .name = nif_name, .arity = FT.params.len, .fptr = nif, .flags = flags };
            };
        }
        const numOfNIFsPerKind = 10;
        const Entries = [NIFs.len + Kinds.len * numOfNIFsPerKind]e.ErlNifFunc;
        fn getEntries() Entries {
            var ret: Entries = undefined;
            @setEvalBranchQuota(8000);
            for (0..NIFs.len) |i| {
                var flags = 0;
                if (NIFs[i].len > 2) {
                    flags = NIFs[i][2].flags;
                }
                ret[i] = KindaNIF(NIFs[i][0], NIFs[i][1], flags).entry;
            }
            for (Kinds, 0..) |k, i| {
                for (0..numOfNIFsPerKind) |j| {
                    ret[NIFs.len + i * numOfNIFsPerKind + j] = k.nifs[j];
                }
            }
            return ret;
        }
        pub const entries = getEntries();
    };
}
