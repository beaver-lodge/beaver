const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
pub const c = prelude.c;
const result = @import("kinda").result;
const kinda = @import("kinda");
const memref = @import("memref.zig");
const e = kinda.erl_nif;
const beam = kinda.beam;

// A constant list of ranks we support with comptime-sized structs.
// This can be easily extended.
const supported_ranks_excluding_zero = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
const supported_ranks_including_zero = .{0} ++ supported_ranks_excluding_zero;

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

// NIF: Get offset
pub fn unranked_memref_descriptor_get_offset(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);
    // Dispatch to cast to the correct type before accessing the field.
    inline for (supported_ranks_including_zero) |R| {
        if (d.rank == R) {
            const D = memref.RankedMemRefDescriptor(R);
            const ranked = @as(*const D, @ptrCast(@alignCast(d.descriptor)));
            return beam.make_i64(env, ranked.offset);
        }
    }
    return error.UnsupportedRank;
}

// Helper function to get either sizes or strides using @field
fn get_ranked_field(env: beam.env, arg: beam.term, comptime field_name: []const u8) !beam.term {
    const d = try UnrankedMemRefDescriptor.resource.fetch(env, arg);
    if (d.rank == 0) {
        return beam.make_term_list(env, &[_]beam.term{});
    }
    // Dispatch to handle the specific RankedDescriptor(R) type.
    inline for (supported_ranks_excluding_zero) |R| {
        if (d.rank == R) {
            const D = memref.RankedMemRefDescriptor(R);
            const ranked = @as(*const D, @ptrCast(@alignCast(d.descriptor)));

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
pub fn unranked_memref_descriptor_get_sizes(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return get_ranked_field(env, args[0], "sizes");
}

// NIF: Get strides as Erlang list
pub fn unranked_memref_descriptor_get_strides(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return get_ranked_field(env, args[0], "strides");
}

fn Deallocator(free: anytype) type {
    return struct {
        fn nif(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const d = try UnrankedMemRefDescriptor.resource.fetch(env, args[0]);
            if (d.rank == 0) {
                const zero_rank = @as(*memref.RankedMemRefDescriptor(0), @ptrCast(@alignCast(d.descriptor)));
                if (zero_rank.allocated) |ptr| {
                    free(ptr);
                    zero_rank.allocated = null;
                    zero_rank.aligned = null;
                    return beam.make_ok(env);
                } else {
                    return beam.make_atom(env, "noop");
                }
            } else {
                inline for (supported_ranks_excluding_zero) |R| {
                    if (d.rank == R) {
                        const D = memref.RankedMemRefDescriptor(R);
                        const ranked = @as(*D, @ptrCast(@alignCast(d.descriptor)));
                        if (ranked.allocated) |ptr| {
                            free(ptr);
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
