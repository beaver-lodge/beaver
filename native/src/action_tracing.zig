const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const result = kinda.result;
const prelude = @import("prelude.zig");
const mlir_capi = @import("mlir_capi.zig");

/// Mirror of the C ABI declared in mlir-c/Beaver/ActionTracing.h.
const MlirBeaverActionTracing = extern struct {
    ptr: ?*anyopaque,
};

const EventsCallback = *const fn (data: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void;

extern fn mlirBeaverActionTracingAttach(
    context: mlir_capi.Context.T,
    filter_json: mlir_capi.StringRef.T,
    location_json: mlir_capi.StringRef.T,
    skip_json: mlir_capi.StringRef.T,
    limit_json: mlir_capi.StringRef.T,
) MlirBeaverActionTracing;

extern fn mlirBeaverActionTracingDrain(
    tracing: MlirBeaverActionTracing,
    callback: EventsCallback,
    user_data: ?*anyopaque,
) bool;

extern fn mlirBeaverActionTracingDetach(tracing: MlirBeaverActionTracing) void;

const TracingState = struct {
    tracing: MlirBeaverActionTracing,
};

fn destroyTracingState(_: beam.env, object: ?*anyopaque) callconv(.c) void {
    const state: *TracingState = @ptrCast(@alignCast(object orelse return));
    if (state.tracing.ptr != null) {
        mlirBeaverActionTracingDetach(state.tracing);
        state.tracing = .{ .ptr = null };
    }
}

const TracingStateResource = kinda.RawResourceType(
    TracingState,
    "Beaver.MLIR.ActionTracing",
    destroyTracingState,
);

fn makeStringRef(env: beam.env, term: beam.term) !mlir_capi.StringRef.T {
    const bin = try beam.get_binary(env, term);
    return .{ .data = bin.data, .length = bin.size };
}

pub fn action_tracing_attach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(env, args[0]);
    const filter_json = try makeStringRef(env, args[1]);
    const location_json = try makeStringRef(env, args[2]);
    const skip_json = try makeStringRef(env, args[3]);
    const limit_json = try makeStringRef(env, args[4]);

    const state = TracingStateResource.alloc() catch
        return error.FailedToAllocateActionTracingState;
    errdefer TracingStateResource.release(state);

    state.tracing =
        mlirBeaverActionTracingAttach(context, filter_json, location_json, skip_json, limit_json);
    if (state.tracing.ptr == null) return error.FailedToAttachActionTracing;

    const term = e.enif_make_resource(env, state);
    TracingStateResource.release(state);
    return term;
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

    _ = mlirBeaverActionTracingDrain(state.tracing, drainCollector, &buffer);
    return beam.make_slice(env, buffer.items);
}

pub fn action_tracing_detach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const state = TracingStateResource.fetch(env, args[0]) catch
        return error.InvalidActionTracingState;
    if (state.tracing.ptr != null) {
        mlirBeaverActionTracingDetach(state.tracing);
        state.tracing = .{ .ptr = null };
    }
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
