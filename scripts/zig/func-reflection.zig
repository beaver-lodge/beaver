const expect = @import("std").testing.expect;
const print = @import("std").debug.print;
const c = @cImport({
    // See https://github.com/ziglang/zig/issues/515
    @cDefine("_NO_CRT_STDIO_INLINE", "1");
    @cInclude("14.h");
});
test "fn reflection" {
    const func_type = @typeInfo(@TypeOf(c.mlirContextEqual));
    comptime var i = 0;
    inline while (i < func_type.Fn.args.len) : (i += 1) {
        const arg_type = func_type.Fn.args[i];
        print("{}\n", .{ arg_type });
    }
    print("{}\n", .{ func_type.Fn.return_type });
}
