const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
pub const c = prelude.c;
const kinda = @import("kinda");
const memref = @import("memref.zig");
const e = kinda.erl_nif;
const beam = kinda.beam;

// Dispatch over supported ranks, including scalar memrefs.
const supported_ranks = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

// Simple Unranked MemRef descriptor
pub const UnrankedMemRefDescriptor = extern struct {
    rank: i64,
    descriptor: *anyopaque, // Points to a rank-specific memref struct
    pub var resource_type: beam.resource_type = undefined;
    pub const resource_name = "Beaver.MLIR.UnrankedMemRefDescriptor";
    const kind = kinda.ResourceKind(@This(), mlir_capi.ElixirNameSpacePrefix ++ "UnrankedMemRefDescriptor");
    const resource = kind.resource;
    pub fn init(rank: i64) !@This() {
        const self = @This(){ .rank = rank, .descriptor = undefined };
        return self;
    }
};

// NIF: Create empty unranked memref
pub fn unranked_memref_descriptor_empty(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const rank: i64 = try beam.get(i64, env, args[0]);
    const d = try UnrankedMemRefDescriptor.init(rank);
    return UnrankedMemRefDescriptor.resource.make(env, d);
}

// NIF: Get rank
pub fn unranked_memref_descriptor_get_rank(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);
    return beam.make_i64(env, d.rank);
}
fn RankedField(comptime name: []const u8) type {
    return struct {
        fn nif(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const d = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);
            inline for (supported_ranks) |rank| {
                if (d.rank == rank) {
                    if (comptime rank == 0 and !std.mem.eql(u8, name, "offset")) return beam.make_term_list(env, &.{});
                    const ranked: *const memref.RankedMemRefDescriptor(rank) = @ptrCast(@alignCast(d.descriptor));
                    if (comptime std.mem.eql(u8, name, "offset")) return beam.make_i64(env, ranked.offset);
                    var terms: [rank]beam.term = undefined;
                    for (@field(ranked, name), 0..) |value, i| terms[i] = beam.make_i64(env, value);
                    return beam.make_term_list(env, &terms);
                }
            }
            return error.UnsupportedRank;
        }
    };
}
pub const unranked_memref_descriptor_get_offset = RankedField("offset").nif;
pub const unranked_memref_descriptor_get_sizes = RankedField("sizes").nif;
pub const unranked_memref_descriptor_get_strides = RankedField("strides").nif;
fn Deallocator(free: anytype) type {
    return struct {
        fn nif(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const d = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);
            inline for (supported_ranks) |rank| {
                if (d.rank == rank) {
                    const ranked: *memref.RankedMemRefDescriptor(rank) = @ptrCast(@alignCast(d.descriptor));
                    const ptr = ranked.allocated orelse return beam.make_atom(env, "noop");
                    free(ptr);
                    ranked.allocated = null;
                    ranked.aligned = null;
                    return beam.make_ok(env);
                }
            }
            return error.UnsupportedRank;
        }
    };
}
pub const unranked_memref_descriptor_deallocate_with_c = Deallocator(std.c.free).nif;
pub const unranked_memref_descriptor_deallocate_with_enif = Deallocator(e.enif_free).nif;

// Export the NIFs
pub const nifs = .{
    prelude.beaverRawNIF(@This(), "unranked_memref_descriptor_empty", 1),
    prelude.beaverRawNIF(@This(), "unranked_memref_descriptor_get_rank", 1),
    prelude.beaverRawNIF(@This(), "unranked_memref_descriptor_get_offset", 1),
    prelude.beaverRawNIF(@This(), "unranked_memref_descriptor_get_sizes", 1),
    prelude.beaverRawNIF(@This(), "unranked_memref_descriptor_get_strides", 1),
    prelude.beaverRawNIF(@This(), "unranked_memref_descriptor_deallocate_with_c", 1),
    prelude.beaverRawNIF(@This(), "unranked_memref_descriptor_deallocate_with_enif", 1),
} ++ UnrankedMemRefDescriptor.kind.nifs;

// Resource type registration
pub fn open_all(env: beam.env) void {
    UnrankedMemRefDescriptor.kind.open_all(env);
}
