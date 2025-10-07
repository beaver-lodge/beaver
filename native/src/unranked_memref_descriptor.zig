const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig").c;
const result = @import("kinda").result;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;

// A constant list of ranks we support with comptime-sized structs.
// This can be easily extended.
const supported_ranks = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

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

// Internal descriptor for rank 0
pub const ZeroRankDescriptor = extern struct {
    allocated: ?*anyopaque = null,
    aligned: ?*anyopaque = null,
    offset: i64,
};

// REFACTORED: Generic descriptor for ranks > 0.
// This is now a function that returns a type, generic over the rank.
pub fn RankedDescriptor(comptime Rank: usize) type {
    return extern struct {
        allocated: ?*anyopaque = null,
        aligned: ?*anyopaque = null,
        offset: i64,
        // The sizes and strides arrays are now embedded directly in the struct.
        sizes: [Rank]i64,
        strides: [Rank]i64,
    };
}

// NIF: Create empty unranked memref
pub fn unranked_memref_empty(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const rank: i64 = try beam.get(i64, env, args[0]);
    const d = try UnrankedMemRefDescriptor.init(rank);
    return UnrankedMemRefDescriptor.resource.make(env, d);
}

// NIF: Get rank
pub fn unranked_memref_get_rank(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);
    return beam.make_i64(env, d.rank);
}

// NIF: Get offset
pub fn unranked_memref_get_offset(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const memref = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);

    if (memref.rank == 0) {
        const zero_rank = @as(*const ZeroRankDescriptor, @ptrCast(@alignCast(memref.descriptor)));
        return beam.make_i64(env, zero_rank.offset);
    } else {
        // Dispatch to cast to the correct type before accessing the field.
        inline for (supported_ranks) |R| {
            if (memref.rank == R) {
                const D = RankedDescriptor(R);
                const ranked = @as(*const D, @ptrCast(@alignCast(memref.descriptor)));
                return beam.make_i64(env, ranked.offset);
            }
        }
        return error.UnsupportedRank;
    }
}

// Helper function to get either sizes or strides using @field
fn get_ranked_field(env: beam.env, memref: UnrankedMemRefDescriptor, comptime field_name: []const u8) !beam.term {
    if (memref.rank == 0) {
        return beam.make_term_list(env, &[_]beam.term{});
    }

    // Dispatch to handle the specific RankedDescriptor(R) type.
    inline for (supported_ranks) |R| {
        if (memref.rank == R) {
            const D = RankedDescriptor(R);
            const ranked = @as(*const D, @ptrCast(@alignCast(memref.descriptor)));

            var terms = try beam.allocator.alloc(beam.term, R);
            defer beam.allocator.free(terms);

            // @field works on the comptime field_name to get either .sizes or .strides.
            const field = @field(ranked, field_name);
            // Iterate over the array (which is now a fixed-size array, not a slice).
            for (field, 0..) |value, i| {
                terms[i] = beam.make_i64(env, value);
            }
            return beam.make_term_list(env, terms);
        }
    }
    return error.UnsupportedRank;
}

// NIF: Get sizes as Erlang list
pub fn unranked_memref_get_sizes(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);
    return get_ranked_field(env, d, "sizes");
}

// NIF: Get strides as Erlang list
pub fn unranked_memref_get_strides(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);
    return get_ranked_field(env, d, "strides");
}

// NIF: Free the externally allocated buffer
pub fn unranked_memref_deallocate(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const memref = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);

    if (memref.rank == 0) {
        const zero_rank = @as(*ZeroRankDescriptor, @ptrCast(@alignCast(memref.descriptor)));
        if (zero_rank.allocated) |ptr| {
            // Memory is assumed to be allocated by a C-compatible allocator.
            std.c.free(ptr);
            zero_rank.allocated = null;
            zero_rank.aligned = null;
            return beam.make_ok(env);
        } else {
            return beam.make_atom(env, "noop");
        }
    } else {
        // Dispatch to cast to the correct type before freeing.
        inline for (supported_ranks) |R| {
            if (memref.rank == R) {
                const D = RankedDescriptor(R);
                const ranked = @as(*D, @ptrCast(@alignCast(memref.descriptor)));
                if (ranked.allocated) |ptr| {
                    std.c.free(ptr);
                    ranked.allocated = null;
                    ranked.aligned = null;
                    return beam.make_ok(env);
                } else {
                    return beam.make_atom(env, "noop");
                }
            }
        }
        return error.UnsupportedRank;
    }
}

// Export the NIFs
pub const nifs = .{
    result.nif("beaver_raw_unranked_memref_descriptor_empty", 1, unranked_memref_empty).entry,
    result.nif("beaver_raw_unranked_memref_descriptor_get_rank", 1, unranked_memref_get_rank).entry,
    result.nif("beaver_raw_unranked_memref_descriptor_get_offset", 1, unranked_memref_get_offset).entry,
    result.nif("beaver_raw_unranked_memref_descriptor_get_sizes", 1, unranked_memref_get_sizes).entry,
    result.nif("beaver_raw_unranked_memref_descriptor_get_strides", 1, unranked_memref_get_strides).entry,
    result.nif("beaver_raw_unranked_memref_descriptor_deallocate", 1, unranked_memref_deallocate).entry,
} ++ UnrankedMemRefDescriptor.kind.nifs;

// Resource type registration
pub fn open_all(env: beam.env) void {
    UnrankedMemRefDescriptor.kind.open_all(env);
}
