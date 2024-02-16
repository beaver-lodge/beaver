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

        fn KindaNIF(comptime cfunction: anytype, comptime FTI: anytype) type {
            return struct {
                inline fn VariadicArgs() type {
                    const P = FTI.params;
                    return switch (P.len) {
                        0 => struct {},
                        1 => struct { P[0].type.? },
                        2 => struct { P[0].type.?, P[1].type.? },
                        3 => struct { P[0].type.?, P[1].type.?, P[2].type.? },
                        4 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.? },
                        5 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.? },
                        6 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.? },
                        7 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.? },
                        8 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.? },
                        9 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.? },
                        10 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.?, P[9].type.? },
                        11 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.?, P[9].type.?, P[10].type.? },
                        12 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.?, P[9].type.?, P[10].type.?, P[11].type.? },
                        13 => struct { P[0].type.?, P[1].type.?, P[2].type.?, P[3].type.?, P[4].type.?, P[5].type.?, P[6].type.?, P[7].type.?, P[8].type.?, P[9].type.?, P[10].type.?, P[11].type.?, P[12].type.? },
                        else => {
                            var buffer: [20]u8 = undefined;
                            const s = std.fmt.bufPrint(&buffer, "{}", .{P.len}) catch unreachable;
                            @compileError("too many args " ++ s ++ ", fn: " ++ @typeName(@TypeOf(cfunction)));
                        },
                    };
                }
                inline fn variadic_call(args: anytype) FTI.return_type.? {
                    const f = cfunction;
                    return switch (FTI.params.len) {
                        0 => f(),
                        1 => f(args[0]),
                        2 => f(args[0], args[1]),
                        3 => f(args[0], args[1], args[2]),
                        4 => f(args[0], args[1], args[2], args[3]),
                        5 => f(args[0], args[1], args[2], args[3], args[4]),
                        6 => f(args[0], args[1], args[2], args[3], args[4], args[5]),
                        7 => f(args[0], args[1], args[2], args[3], args[4], args[5], args[6]),
                        8 => f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
                        9 => f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]),
                        10 => f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]),
                        11 => f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]),
                        12 => f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]),
                        13 => f(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]),
                        else => @compileError("too many args"),
                    };
                }
                fn nif(env: beam.env, _: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
                    var c_args: VariadicArgs() = undefined;
                    inline for (FTI.params, args, 0..) |p, arg, i| {
                        const ArgKind = getKind(p.type.?);
                        c_args[i] = ArgKind.resource.fetch(env, arg) catch return beam.make_error_binary(env, "fail to fetch arg resource");
                    }
                    const rt = FTI.return_type.?;
                    if (rt == void) {
                        variadic_call(c_args);
                        return beam.make_ok(env);
                    } else {
                        const RetKind = getKind(rt);
                        return RetKind.resource.make(env, variadic_call(c_args)) catch return beam.make_error_binary(env, "fail to make resource for return type");
                    }
                }
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
                const FTI = @typeInfo(@TypeOf(NIFs[i][0])).Fn;
                const cfunction = NIFs[i][0];
                const nif = KindaNIF(cfunction, @typeInfo(@TypeOf(cfunction)).Fn).nif;
                const entry = e.ErlNifFunc{ .name = NIFs[i][1], .arity = FTI.params.len, .fptr = nif, .flags = flags };
                ret[i] = entry;
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
