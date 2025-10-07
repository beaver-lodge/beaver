const std = @import("std");
const mem = @import("std").mem;
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig").c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const result = kinda.result;

pub fn memref_type_get_strides_and_offset(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const t = try mlir_capi.Type.resource.fetch(env, args[0]);
    const rank = c.mlirShapedTypeGetRank(t);
    if (rank < 0) {
        return error.UnrankedMemref;
    }
    const strides_slice: []i64 = try beam.allocator.alloc(i64, @intCast(rank));
    defer beam.allocator.free(strides_slice);

    var offset: i64 = undefined;

    const res = c.mlirMemRefTypeGetStridesAndOffset(t, strides_slice.ptr, &offset);

    if (!c.beaverLogicalResultIsSuccess(res)) {
        return error.LayoutNotStrided;
    }

    var strides_terms: []beam.term = try beam.allocator.alloc(beam.term, @intCast(rank));
    defer beam.allocator.free(strides_terms);

    for (strides_slice, 0..) |s, i| {
        strides_terms[i] = try beam.make(i64, env, s);
    }

    const strides_list = beam.make_term_list(env, strides_terms);
    const offset_term = try beam.make(i64, env, offset);

    var ret_slice: []beam.term = try beam.allocator.alloc(beam.term, 2);
    defer beam.allocator.free(ret_slice);
    ret_slice[0] = strides_list;
    ret_slice[1] = offset_term;
    return beam.make_tuple(env, ret_slice);
}

pub const nifs = .{
    result.nif("beaver_raw_memref_type_get_strides_and_offset", 1, memref_type_get_strides_and_offset).entry,
};
