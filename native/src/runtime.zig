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
    var b: beam.binary = undefined;
    if (e.enif_inspect_binary(env, term, &b) == 0) {
        @panic("failed to inspect binary");
    }
    d.* = BinaryMemRefDescriptor{ .allocated = null, .aligned = b.data, .offset = 0, .sizes = .{@intCast(b.size)}, .strides = .{1} };
}
pub fn __decl__inspect_binary_as_memref(_: beam.env, _: beam.term) callconv(.c) BinaryMemRefDescriptor {
    @panic("call inspect_binary_as_memref for correct ABI");
}
// On Windows, translate-c cannot infer concrete signatures for the
// enif_make_list/tupleN convenience macros (they become generic inline
// functions), so provide typed stand-ins for signature introspection.
pub fn __decl__enif_make_list1(env: beam.env, e1: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 1, e1);
}
pub fn __decl__enif_make_list2(env: beam.env, e1: beam.term, e2: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 2, e1, e2);
}
pub fn __decl__enif_make_list3(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 3, e1, e2, e3);
}
pub fn __decl__enif_make_list4(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 4, e1, e2, e3, e4);
}
pub fn __decl__enif_make_list5(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 5, e1, e2, e3, e4, e5);
}
pub fn __decl__enif_make_list6(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term, e6: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 6, e1, e2, e3, e4, e5, e6);
}
pub fn __decl__enif_make_list7(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term, e6: beam.term, e7: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 7, e1, e2, e3, e4, e5, e6, e7);
}
pub fn __decl__enif_make_list8(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term, e6: beam.term, e7: beam.term, e8: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 8, e1, e2, e3, e4, e5, e6, e7, e8);
}
pub fn __decl__enif_make_list9(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term, e6: beam.term, e7: beam.term, e8: beam.term, e9: beam.term) callconv(.c) beam.term {
    return e.enif_make_list(env, 9, e1, e2, e3, e4, e5, e6, e7, e8, e9);
}
pub fn __decl__enif_make_tuple1(env: beam.env, e1: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 1, e1);
}
pub fn __decl__enif_make_tuple2(env: beam.env, e1: beam.term, e2: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 2, e1, e2);
}
pub fn __decl__enif_make_tuple3(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 3, e1, e2, e3);
}
pub fn __decl__enif_make_tuple4(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 4, e1, e2, e3, e4);
}
pub fn __decl__enif_make_tuple5(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 5, e1, e2, e3, e4, e5);
}
pub fn __decl__enif_make_tuple6(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term, e6: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 6, e1, e2, e3, e4, e5, e6);
}
pub fn __decl__enif_make_tuple7(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term, e6: beam.term, e7: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 7, e1, e2, e3, e4, e5, e6, e7);
}
pub fn __decl__enif_make_tuple8(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term, e6: beam.term, e7: beam.term, e8: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 8, e1, e2, e3, e4, e5, e6, e7, e8);
}
pub fn __decl__enif_make_tuple9(env: beam.env, e1: beam.term, e2: beam.term, e3: beam.term, e4: beam.term, e5: beam.term, e6: beam.term, e7: beam.term, e8: beam.term, e9: beam.term) callconv(.c) beam.term {
    return e.enif_make_tuple(env, 9, e1, e2, e3, e4, e5, e6, e7, e8, e9);
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
