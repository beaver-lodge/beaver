const debug_print = @import("std").debug.print;
pub fn print_i32(i: i32) callconv(.C) void {
    debug_print("{}", .{i});
}
pub fn print_u32(i: u32) callconv(.C) void {
    debug_print("{}", .{i});
}
pub fn print_i64(i: i64) callconv(.C) void {
    debug_print("{}", .{i});
}
pub fn print_u64(i: u64) callconv(.C) void {
    debug_print("{}", .{i});
}
pub fn print_f32(f: f32) callconv(.C) void {
    debug_print("{}", .{f});
}
pub fn print_f64(f: f64) callconv(.C) void {
    debug_print("{}", .{f});
}
pub fn print_open() callconv(.C) void {
    debug_print("( ", .{});
}
pub fn print_close() callconv(.C) void {
    debug_print(" )", .{});
}
pub fn print_comma() callconv(.C) void {
    debug_print(", ", .{});
}
pub fn print_newline() callconv(.C) void {
    debug_print("\n", .{});
}
const e = @import("erl_nif");
pub usingnamespace e;
