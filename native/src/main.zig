const std = @import("std");
const mem = @import("std").mem;
const testing = std.testing;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const mlir_capi = @import("mlir_capi.zig");
const enif_support = @import("enif_support.zig");
const diagnostic = @import("diagnostic.zig");
const pass = @import("pass.zig");
const registry = @import("registry.zig");
const pointer = @import("pointer.zig");
const string_ref = @import("string_ref.zig");
const memref = @import("memref.zig");
const unranked_memref_descriptor = @import("unranked_memref_descriptor.zig");
const logical_mutex = @import("logical_mutex.zig");

const handwritten_nifs = @import("wrapper.zig").nif_entries ++ mlir_capi.EntriesOfKinds ++ pass.nifs ++ registry.nifs ++ string_ref.nifs ++ diagnostic.nifs ++ pointer.nifs ++ memref.nifs ++ enif_support.nifs ++ logical_mutex.nifs ++ unranked_memref_descriptor.nifs;

const num_nifs = handwritten_nifs.len;
export var nifs: [num_nifs]e.ErlNifFunc = handwritten_nifs;

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
    pass.register_all_passes();
    kinda.open_internal_resource_types(env);
    kinda.Internal.OpaqueStruct.open_all(env);
    mlir_capi.open_all(env);
    unranked_memref_descriptor.open_all(env);
    logical_mutex.open_all(env);
    return 0;
}

const entry = e.ErlNifEntry{
    .major = 2,
    .minor = 16,
    .name = mlir_capi.root_module,
    .num_of_funcs = num_nifs,
    .funcs = &(nifs[0]),
    .load = nif_load,
    .reload = null, // currently unsupported
    .upgrade = null, // currently unsupported
    .unload = null, // currently unsupported
    .vm_variant = "beam.vanilla",
    .options = 1,
    .sizeof_ErlNifResourceTypeInit = @sizeOf(e.ErlNifResourceTypeInit),
    .min_erts = "erts-13.0",
};

export fn nif_init() *const e.ErlNifEntry {
    return &entry;
}
