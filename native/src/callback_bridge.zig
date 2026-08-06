const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const mlir_capi = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");

const SpeculatabilityDispatcher = kinda.callback_runtime.Dispatcher(.{"get_speculatability"});

var speculatability_state_type: beam.resource_type = undefined;

const SpeculatabilityState = struct {
    dispatcher: *SpeculatabilityDispatcher,

    fn construct(_: ?*anyopaque) callconv(.c) void {}

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        e.enif_release_resource(self);
    }

    fn getSpeculatability(
        operation: mlir_capi.Operation.T,
        user_data: ?*anyopaque,
    ) callconv(.c) c.MlirSpeculatability {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.MlirSpeculatabilityNotSpeculatable));
        const environment = e.enif_alloc_env() orelse
            return c.MlirSpeculatabilityNotSpeculatable;
        const operation_term = kinda.callback_adapter.handle(
            mlir_capi.Operation,
            environment,
            operation,
        ) catch {
            e.enif_free_env(environment);
            return c.MlirSpeculatabilityNotSpeculatable;
        };
        const response = self.dispatcher.invoke(
            "get_speculatability",
            environment,
            .{operation_term},
        ) catch return c.MlirSpeculatabilityNotSpeculatable;
        return kinda.callback_adapter.enumResult(
            c.MlirSpeculatability,
            response,
            c.MlirSpeculatabilityNotSpeculatable,
        );
    }
};

fn destroySpeculatabilityState(_: beam.env, object: ?*anyopaque) callconv(.c) void {
    const self: *SpeculatabilityState = @ptrCast(@alignCast(object orelse return));
    self.dispatcher.deinit();
}

fn attachSpeculatabilityFallback(
    environment: beam.env,
    _: c_int,
    args: [*c]const beam.term,
) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
    const dispatcher = try SpeculatabilityDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try beam.get_u64(environment, args[3]),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    dispatcher.setCallback("get_speculatability", args[2]);

    const memory = e.enif_alloc_resource(
        speculatability_state_type,
        @sizeOf(SpeculatabilityState),
    ) orelse return error.FailedToAllocateSpeculatabilityState;
    const state: *SpeculatabilityState = @ptrCast(@alignCast(memory));
    errdefer e.enif_release_resource(state);
    state.* = .{ .dispatcher = dispatcher };
    dispatcher_owned = false;
    const callbacks = c.MlirConditionallySpeculatableOpInterfaceCallbacks{
        .construct = SpeculatabilityState.construct,
        .destruct = SpeculatabilityState.destruct,
        .getSpeculatability = SpeculatabilityState.getSpeculatability,
        .userData = state,
    };
    c.mlirConditionallySpeculatableOpInterfaceAttachFallbackModel(
        context,
        try string_ref.get_binary_as_string_ref(environment, args[1]),
        callbacks,
    );
    return dispatcher.copyId(environment);
}

const SpeculatabilityWorker = struct {
    operation: mlir_capi.Operation.T,
    recipient: beam.pid,

    fn run(user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        defer std.heap.smp_allocator.destroy(self);
        const environment = e.enif_alloc_env() orelse return;
        defer e.enif_free_env(environment);
        const value = c.mlirConditionallySpeculatableOpInterfaceGetSpeculatability(self.operation);
        var terms = [_]beam.term{
            beam.make_atom(environment, "speculatability_done"),
            beam.make_c_int(environment, @intCast(value)),
        };
        _ = beam.send_advanced(null, self.recipient, environment, beam.make_tuple(environment, &terms));
    }
};

fn querySpeculatabilityAsync(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const operation = try mlir_capi.Operation.resource.fetch(environment, args[0]);
    const worker = try std.heap.smp_allocator.create(SpeculatabilityWorker);
    errdefer std.heap.smp_allocator.destroy(worker);
    worker.* = .{ .operation = operation, .recipient = try beam.self(environment) };
    const context = c.mlirOperationGetContext(operation);
    if (!c.beaverContextAddWork(context, SpeculatabilityWorker.run, worker)) {
        return error.ContextMultithreadingDisabled;
    }
    return beam.make_ok(environment);
}

pub fn open(environment: beam.env) void {
    speculatability_state_type = e.enif_open_resource_type(
        environment,
        null,
        "Beaver.MLIR.ConditionallySpeculatable.FallbackState",
        destroySpeculatabilityState,
        e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER,
        null,
    );
    if (speculatability_state_type == null)
        @panic("failed to open speculatability fallback state resource");
}

pub const nifs = .{
    prelude.beaverRawNIF(@This(), "conditionally_speculatable_attach_fallback_model", 4),
    prelude.beaverRawNIF(@This(), "conditionally_speculatable_query_async", 1),
    kinda.callback_runtime.ReplyToken.codeNif("beaver_raw_callback_reply_code"),
};

pub const conditionally_speculatable_attach_fallback_model = attachSpeculatabilityFallback;
pub const conditionally_speculatable_query_async = querySpeculatabilityAsync;
