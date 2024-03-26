pub fn beaver_debug_print_i32(i: i32) callconv(.C) void {
    @import("std").debug.print("debug print: {}\n", .{i});
}
const e = @import("erl_nif");
pub usingnamespace e;
