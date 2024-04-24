const beam = @import("beam");
const e = @import("erl_nif");

pub fn nif(comptime name: [*c]const u8, comptime arity: usize, comptime f: anytype) type {
    return struct {
        fn exported(env: beam.env, n: c_int, args: [*c]const beam.term) callconv(.C) beam.term {
            var ret: beam.term = undefined;
            if (f(env, n, args)) |r| {
                ret = r;
            } else |err| {
                ret = beam.make_error_term(env, beam.make_slice(env, @errorName(err)));
            }
            return ret;
        }
        pub const entry = e.ErlNifFunc{ .name = name, .arity = arity, .fptr = exported, .flags = 0 };
    };
}
