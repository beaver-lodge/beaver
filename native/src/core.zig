const std = @import("std");
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
const action_tracing = @import("action_tracing.zig");
const triton = @import("triton.zig");
const llvm_ir = @import("llvm_ir.zig");
const infer_type = @import("infer_type.zig");
const capi_registry = @import("capi_registry.zig");
const logical_result = @import("logical_result.zig");

const callback_nifs = .{kinda.callback_runtime.ReplyToken.nif("beaver_raw_callback_reply")};

pub const nifs = capi_registry.nifs ++
    logical_result.nifs ++
    mlir_capi.EntriesOfKinds ++
    pass.nifs ++
    registry.nifs ++
    string_ref.nifs ++
    diagnostic.nifs ++
    pointer.nifs ++
    memref.nifs ++
    enif_support.nifs ++
    callback_nifs ++
    unranked_memref_descriptor.nifs ++
    value.nifs ++
    action_tracing.nifs ++
    llvm_ir.nifs ++
    infer_type.nifs ++
    triton.nifs;

export const core_nifs: [nifs.len]e.ErlNifFunc = nifs;
export const core_nifs_len: usize = nifs.len;

pub fn open_all(env: beam.env) void {
    kinda.open_internal_resource_types(env);
    kinda.Internal.OpaqueStruct.open_all(env);
    mlir_capi.open_all(env);
    kinda.callback_runtime.ReplyToken.open(env);
    unranked_memref_descriptor.open_all(env);
    action_tracing.open(env);
}

pub fn register_all_passes() void {
    pass.register_all_passes();
}

/// Returns the resource type handle opened by this partition for a resource
/// name. The leaf partitions call this during `nif_load` instead of reopening
/// the same resource types: opening the same name twice within one load
/// creates distinct resource types, so handles must be shared explicitly.
export fn core_resource_type_by_name(name: [*:0]const u8) ?*anyopaque {
    const name_slice = std.mem.span(name);
    inline for (mlir_capi.resourceSlots) |slot| {
        if (std.mem.eql(u8, name_slice, slot.name)) {
            return @ptrCast(slot.t.*);
        }
    }
    return null;
}

export fn core_open_all(env: beam.env) void {
    open_all(env);
}

export fn core_register_all_passes() void {
    register_all_passes();
}
