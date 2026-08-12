const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;

const mlir_capi = @import("mlir_capi.zig");

extern fn core_resource_type_by_name(name: [*:0]const u8) ?*anyopaque;

/// Copies the resource type handles opened by the core partition into this
/// partition's kind instances. Resource type handles cannot be reopened during
/// the same NIF load, so every partition that fetches or creates kind terms
/// must share the core-opened handles this way.
pub fn syncResourceTypes() void {
    inline for (mlir_capi.resourceSlots) |slot| {
        const handle: ?*e.ErlNifResourceType = @ptrCast(core_resource_type_by_name(@ptrCast(slot.name.ptr)));
        slot.t.* = handle;
    }
}
