const std = @import("std");
const mem = @import("std").mem;
const testing = std.testing;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const mlir_capi = @import("mlir_capi.zig");
const enif_support = @import("enif_support.zig");
const prelude = @import("prelude.zig");
const diagnostic = @import("diagnostic.zig");
const pass = @import("pass.zig");
const registry = @import("registry.zig");
const pointer = @import("pointer.zig");
const string_ref = @import("string_ref.zig");
const memref = @import("memref.zig");
const unranked_memref_descriptor = @import("unranked_memref_descriptor.zig");
const value = @import("value.zig");

const rewrite_pattern = @import("rewrite_pattern.zig");
const capi_registry = @import("capi_registry.zig");
const callback_nifs = .{kinda.callback_runtime.ReplyToken.nif("beaver_raw_callback_reply")};
const handwritten_nifs = capi_registry.nifs ++ mlir_capi.EntriesOfKinds ++ pass.nifs ++ registry.nifs ++ string_ref.nifs ++ diagnostic.nifs ++ pointer.nifs ++ memref.nifs ++ enif_support.nifs ++ callback_nifs ++ unranked_memref_descriptor.nifs ++ rewrite_pattern.nifs ++ value.nifs;

const num_nifs = handwritten_nifs.len;
export var nifs: [num_nifs]e.ErlNifFunc = handwritten_nifs;

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
    pass.register_all_passes();
    kinda.open_internal_resource_types(env);
    kinda.Internal.OpaqueStruct.open_all(env);
    mlir_capi.open_all(env);
    unranked_memref_descriptor.open_all(env);
    kinda.callback_runtime.ReplyToken.open(env);
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
