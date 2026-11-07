const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const prelude = @import("prelude.zig");
const c = prelude.c;
const mlir_capi = @import("mlir_capi.zig");
const makeStringRef = @import("string_ref.zig").get_binary_as_string_ref;

const TracingState = struct { tracing: c.MlirBeaverActionTracing };

fn destroyTracingState(_: beam.env, object: ?*anyopaque) callconv(.c) void {
    const state: *TracingState = @ptrCast(@alignCast(object orelse return));
    if (state.tracing.ptr != null) {
        c.mlirBeaverActionTracingDetach(state.tracing);
        state.tracing = .{ .ptr = null };
    }
}

const TracingStateResource = kinda.RawResourceType(TracingState, "Beaver.MLIR.ActionTracing", destroyTracingState);

pub fn action_tracing_attach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(env, args[0]);
    const filter_json = try makeStringRef(env, args[1]);
    const location_json = try makeStringRef(env, args[2]);
    const skip_json = try makeStringRef(env, args[3]);
    const limit_json = try makeStringRef(env, args[4]);

    const tracing = c.mlirBeaverActionTracingAttach(context, filter_json, location_json, skip_json, limit_json);
    if (tracing.ptr == null) return error.FailedToAttachActionTracing;
    return TracingStateResource.make(env, .{ .tracing = tracing }) catch {
        c.mlirBeaverActionTracingDetach(tracing);
        return error.FailedToAllocateActionTracingState;
    };
}

fn drainCollector(data: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    const output: *std.array_list.Managed(u8) = @ptrCast(@alignCast(user_data orelse return));
    if (data) |slice| output.appendSlice(std.mem.span(slice)) catch {};
}

pub fn action_tracing_drain(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const state = TracingStateResource.fetch(env, args[0]) catch
        return error.InvalidActionTracingState;

    var buffer = std.array_list.Managed(u8).init(beam.allocator);
    defer buffer.deinit();

    _ = c.mlirBeaverActionTracingDrain(state.tracing, drainCollector, &buffer);
    return beam.make_slice(env, buffer.items);
}

pub fn action_tracing_detach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const state = TracingStateResource.fetch(env, args[0]) catch
        return error.InvalidActionTracingState;
    destroyTracingState(env, state);
    return beam.make_ok(env);
}

pub fn open(environment: beam.env) void {
    TracingStateResource.open(environment);
}

pub const nifs = .{
    prelude.beaverRawNIF(@This(), "action_tracing_attach", 5),
    prelude.beaverRawNIF(@This(), "action_tracing_drain", 1),
    prelude.beaverRawNIF(@This(), "action_tracing_detach", 1),
};
