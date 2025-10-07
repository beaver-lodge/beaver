const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const mlir_capi = @import("mlir_capi.zig");

const debug_print = @import("std").debug.print;
pub fn print_i32(i: i32) callconv(.c) void {
    debug_print("{}", .{i});
}
pub fn print_u32(i: u32) callconv(.c) void {
    debug_print("{}", .{i});
}
pub fn print_i64(i: i64) callconv(.c) void {
    debug_print("{}", .{i});
}
pub fn print_u64(i: u64) callconv(.c) void {
    debug_print("{}", .{i});
}
pub fn print_f32(f: f32) callconv(.c) void {
    debug_print("{}", .{f});
}
pub fn print_f64(f: f64) callconv(.c) void {
    debug_print("{}", .{f});
}
pub fn print_open() callconv(.c) void {
    debug_print("( ", .{});
}
pub fn print_close() callconv(.c) void {
    debug_print(" )", .{});
}
pub fn print_comma() callconv(.c) void {
    debug_print(", ", .{});
}
pub fn print_newline() callconv(.c) void {
    debug_print("\n", .{});
}

pub const BinaryMemRefDescriptor = @import("unranked_memref_descriptor.zig").RankedDescriptor(1);
pub const BinaryMemRefType = "memref<?xi8>";
// Due to the change of function signature when MemRef is converted to LLVM, we can't implement this function with MLIR CAPI only.
// One other way is to implement a conversion pass for LLVMConversionTarget in C++.
const Ptr = *u8;
pub fn ptr_to_memref(d: *BinaryMemRefDescriptor, ptr: Ptr, size: usize) callconv(.c) void {
    d.* = BinaryMemRefDescriptor{ .allocated = ptr, .aligned = ptr, .offset = 0, .sizes = .{@intCast(size)}, .strides = .{1} };
}
pub fn __decl__ptr_to_memref(_: *u8, _: usize) callconv(.c) BinaryMemRefDescriptor {
    @panic("call ptr_to_memref for correct ABI");
}
pub const exported = .{ "print_i32", "print_u32", "print_i64", "print_u64", "print_f32", "print_f64", "print_open", "print_close", "print_comma", "print_newline", "ptr_to_memref" };
