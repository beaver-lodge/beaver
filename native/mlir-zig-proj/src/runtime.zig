const e = @import("erl_nif");
const beam = @import("beam");
const m = @import("memref.zig");
const mlir_capi = @import("mlir_capi.zig");

pub usingnamespace e;
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

pub const BinaryMemRefDescriptor = m.MemRefDescriptor(mlir_capi.U8, 1);
pub const BinaryMemRefType = "memref<?xi8>";
pub fn __decl__inspect_binary_as_memref(_: beam.env, _: beam.term) callconv(.C) BinaryMemRefDescriptor {
    @panic("call inspect_binary_as_memref for correct ABI");
}
pub fn inspect_binary_as_memref(d: *BinaryMemRefDescriptor, env: beam.env, term: beam.term) callconv(.C) void {
    const b: beam.binary = beam.get_binary(env, term) catch @panic("expected binary");
    d.* = BinaryMemRefDescriptor{ .allocated = b.data, .aligned = b.data, .offset = 0, .sizes = .{@intCast(b.size)}, .strides = .{1} };
}
pub const exported = .{ "print_i32", "print_u32", "print_i64", "print_u64", "print_f32", "print_f64", "print_open", "print_close", "print_comma", "print_newline", "inspect_binary_as_memref" };
