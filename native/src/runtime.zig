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

pub const BinaryMemRefDescriptor = @import("memref.zig").RankedMemRefDescriptor(1);
pub const BinaryMemRefType = "memref<?xi8>";
pub const BinaryStructLLVMType = "!llvm.struct<(i64, ptr)>";

// `__decl__` function are added due to the change of function signature during MemRef-to-LLVM translation
pub fn make_new_binary_as_memref(d: *BinaryMemRefDescriptor, env: beam.env, size: usize, term_ptr: [*c]beam.term) callconv(.c) void {
    const ptr = e.enif_make_new_binary(env, size, term_ptr);
    d.* = BinaryMemRefDescriptor{ .allocated = ptr, .aligned = ptr, .offset = 0, .sizes = .{@intCast(size)}, .strides = .{1} };
}
pub fn __decl__make_new_binary_as_memref(_: beam.env, _: usize, _: [*c]beam.term) callconv(.c) BinaryMemRefDescriptor {
    @panic("call make_new_binary_as_memref for correct ABI");
}
pub fn inspect_binary_as_memref(d: *BinaryMemRefDescriptor, env: beam.env, term: beam.term) callconv(.c) void {
    var b : beam.binary = undefined;
    if (e.enif_inspect_binary(env, term, &b) == 0) {
       @panic("failed to inspect binary");
    }
    d.* = BinaryMemRefDescriptor{ .allocated = null, .aligned = b.data, .offset = 0, .sizes = .{@intCast(b.size)}, .strides = .{1} };
}
pub fn __decl__inspect_binary_as_memref(_: beam.env, _: beam.term) callconv(.c) BinaryMemRefDescriptor {
    @panic("call inspect_binary_as_memref for correct ABI");
}
pub const exported = .{
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
    "inspect_binary_as_memref",
    "make_new_binary_as_memref",
};
