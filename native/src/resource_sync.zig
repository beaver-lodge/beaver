const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;

const mlir_capi = @import("mlir_capi.zig");

extern fn core_resource_type_by_name(name: [*:0]const u8) ?*anyopaque;

fn lookup(name: [*:0]const u8) ?*e.ErlNifResourceType {
    const handle = core_resource_type_by_name(name) orelse return null;
    return @ptrCast(handle);
}

fn syncKind(comptime k: type) void {
    k.resource.t = lookup(k.resource.name);
    k.Ptr.resource.t = lookup(k.Ptr.resource.name);
    k.Array.resource.t = lookup(k.Array.resource.name);
}

/// Copies the resource type handles opened by the core partition into this
/// partition's kind instances. Resource type handles cannot be reopened during
/// the same NIF load, so every partition that fetches or creates kind terms
/// must share the core-opened handles this way.
pub fn syncResourceTypes() void {
    inline for (mlir_capi.allKinds) |k| {
        syncKind(k);
    }
    syncKind(kinda.Internal.OpaquePtr);
    syncKind(kinda.Internal.OpaqueArray);
    syncKind(kinda.Internal.USize);
    syncKind(kinda.Internal.OpaqueStruct);
    kinda.callback_runtime.ReplyToken.resource_type = lookup(kinda.callback_runtime.ReplyToken.resource_name);
}
