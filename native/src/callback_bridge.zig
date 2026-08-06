const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const mlir_capi = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");

const TypeConverterDispatcher = kinda.callback_runtime.Dispatcher(.{"convert_type"});
const SpeculatabilityDispatcher = kinda.callback_runtime.Dispatcher(.{"get_speculatability"});

var type_converter_registration_type: beam.resource_type = undefined;
var speculatability_state_type: beam.resource_type = undefined;

const TypeConverterRegistration = struct {
    converter: mlir_capi.TypeConverter.T,
    dispatcher: *TypeConverterDispatcher,
    mutex: std.Io.Mutex = .init,
    closed: std.atomic.Value(bool) = .init(false),

    fn close(self: *@This()) bool {
        const io = std.Options.debug_io;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        if (self.closed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return false;
        c.mlirTypeConverterDestroy(self.converter);
        self.dispatcher.deinit();
        return true;
    }
};

fn destroyTypeConverterRegistration(_: beam.env, object: ?*anyopaque) callconv(.c) void {
    const registration: *TypeConverterRegistration = @ptrCast(@alignCast(object orelse return));
    _ = registration.close();
}

fn fetchTypeConverterRegistration(environment: beam.env, term: beam.term) !*TypeConverterRegistration {
    return beam.fetch_resource_ptr(
        *TypeConverterRegistration,
        environment,
        type_converter_registration_type,
        term,
    );
}

fn typeConversionCallback(
    type_: mlir_capi.Type.T,
    converted_type: [*c]mlir_capi.Type.T,
    user_data: ?*anyopaque,
) callconv(.c) c.MlirTypeConverterConversionStatus {
    const registration: *TypeConverterRegistration = @ptrCast(@alignCast(user_data orelse
        return c.MlirTypeConverterConversionStatusFailure));
    if (registration.closed.load(.acquire)) return c.MlirTypeConverterConversionStatusFailure;

    const message_env = e.enif_alloc_env() orelse
        return c.MlirTypeConverterConversionStatusFailure;
    const type_term = kinda.callback_adapter.handle(mlir_capi.Type, message_env, type_) catch {
        e.enif_free_env(message_env);
        return c.MlirTypeConverterConversionStatusFailure;
    };
    const response = registration.dispatcher.invoke("convert_type", message_env, .{type_term}) catch
        return c.MlirTypeConverterConversionStatusFailure;
    const status = kinda.callback_adapter.enumResult(
        c.MlirTypeConverterConversionStatus,
        response,
        c.MlirTypeConverterConversionStatusFailure,
    );

    if (status == c.MlirTypeConverterConversionStatusSuccess) {
        const projection = kinda.callback_adapter.projection(response) orelse
            return c.MlirTypeConverterConversionStatusFailure;
        converted_type.* = .{ .ptr = @ptrFromInt(projection) };
    }
    return status;
}

fn createTypeConverter(
    environment: beam.env,
    _: c_int,
    args: [*c]const beam.term,
) !beam.term {
    const dispatcher = try TypeConverterDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try beam.get_u64(environment, args[1]),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    dispatcher.setCallback("convert_type", args[0]);

    const memory = e.enif_alloc_resource(
        type_converter_registration_type,
        @sizeOf(TypeConverterRegistration),
    ) orelse return error.FailedToAllocateTypeConverterRegistration;
    const registration: *TypeConverterRegistration = @ptrCast(@alignCast(memory));
    registration.* = .{
        .converter = c.mlirTypeConverterCreate(),
        .dispatcher = dispatcher,
    };
    dispatcher_owned = false;
    c.mlirTypeConverterAddConversion(
        registration.converter,
        typeConversionCallback,
        registration,
    );

    const registration_term = e.enif_make_resource(environment, memory);
    e.enif_release_resource(memory);
    var terms = [_]beam.term{
        beam.make_atom(environment, "callback_type_converter"),
        try mlir_capi.TypeConverter.resource.make_kind(environment, registration.converter),
        registration_term,
    };
    return beam.make_tuple(environment, &terms);
}

fn replyTypeConversion(
    environment: beam.env,
    _: c_int,
    args: [*c]const beam.term,
) !beam.term {
    const success = try beam.get_bool(environment, args[1]);
    const code = try beam.get_i64(environment, args[2]);
    var projection: usize = 0;
    if (success and code == c.MlirTypeConverterConversionStatusSuccess) {
        const converted_type = try mlir_capi.Type.resource.fetch(environment, args[3]);
        projection = @intFromPtr(converted_type.ptr);
    }
    const accepted = try kinda.callback_runtime.ReplyToken.complete(
        environment,
        args[0],
        .replied,
        success,
        code,
        projection,
    );
    return beam.make_atom(environment, if (accepted) "ok" else "stale");
}

const TypeConverterWorker = struct {
    registration: *TypeConverterRegistration,
    recipient: beam.pid,
    type_: mlir_capi.Type.T,
    destroy: bool = false,

    fn run(self: *@This()) void {
        const registration = self.registration;
        defer std.heap.smp_allocator.destroy(self);
        defer e.enif_release_resource(registration);

        const environment = e.enif_alloc_env() orelse return;
        defer e.enif_free_env(environment);

        if (self.destroy) {
            const id = registration.dispatcher.copyId(environment);
            _ = registration.close();
            var terms = [_]beam.term{
                beam.make_atom(environment, "type_converter_destroyed"),
                id,
            };
            _ = beam.send_advanced(null, self.recipient, environment, beam.make_tuple(environment, &terms));
            return;
        }

        const io = std.Options.debug_io;
        registration.mutex.lockUncancelable(io);
        if (registration.closed.load(.acquire)) {
            registration.mutex.unlock(io);
            var terms = [_]beam.term{
                beam.make_atom(environment, "type_converter_error"),
                beam.make_atom(environment, "registration_closed"),
            };
            _ = beam.send_advanced(null, self.recipient, environment, beam.make_tuple(environment, &terms));
            return;
        }

        const converted = c.mlirTypeConverterConvertType(registration.converter, self.type_);
        registration.mutex.unlock(io);
        var terms = [_]beam.term{
            beam.make_atom(environment, "type_converter_done"),
            registration.dispatcher.copyId(environment),
            if (c.mlirTypeIsNull(converted))
                beam.make_nil(environment)
            else
                mlir_capi.Type.resource.make_kind(environment, converted) catch beam.make_nil(environment),
        };
        _ = beam.send_advanced(null, self.recipient, environment, beam.make_tuple(environment, &terms));
    }
};

fn spawnTypeConverterWorker(
    environment: beam.env,
    registration: *TypeConverterRegistration,
    type_: mlir_capi.Type.T,
    destroy: bool,
) !beam.term {
    const worker = try std.heap.smp_allocator.create(TypeConverterWorker);
    errdefer std.heap.smp_allocator.destroy(worker);
    worker.* = .{
        .registration = registration,
        .recipient = try beam.self(environment),
        .type_ = type_,
        .destroy = destroy,
    };
    e.enif_keep_resource(registration);
    errdefer e.enif_release_resource(registration);
    const thread = try std.Thread.spawn(.{}, TypeConverterWorker.run, .{worker});
    thread.detach();
    return registration.dispatcher.copyId(environment);
}

fn convertTypeAsync(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return spawnTypeConverterWorker(
        environment,
        try fetchTypeConverterRegistration(environment, args[0]),
        try mlir_capi.Type.resource.fetch(environment, args[1]),
        false,
    );
}

fn destroyTypeConverterAsync(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return spawnTypeConverterWorker(
        environment,
        try fetchTypeConverterRegistration(environment, args[0]),
        .{ .ptr = null },
        true,
    );
}

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
    type_converter_registration_type = e.enif_open_resource_type(
        environment,
        null,
        "Beaver.MLIR.TypeConverter.CallbackRegistration",
        destroyTypeConverterRegistration,
        e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER,
        null,
    );
    if (type_converter_registration_type == null)
        @panic("failed to open type converter callback registration resource");

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
    prelude.beaverRawNIF(@This(), "type_converter_create_callback", 2),
    prelude.beaverRawNIF(@This(), "type_converter_reply_callback", 4),
    prelude.beaverRawNIF(@This(), "type_converter_convert_async", 2),
    prelude.beaverRawNIF(@This(), "type_converter_destroy_async", 1),
    prelude.beaverRawNIF(@This(), "conditionally_speculatable_attach_fallback_model", 4),
    prelude.beaverRawNIF(@This(), "conditionally_speculatable_query_async", 1),
    kinda.callback_runtime.ReplyToken.codeNif("beaver_raw_callback_reply_code"),
};

pub const type_converter_create_callback = createTypeConverter;
pub const type_converter_reply_callback = replyTypeConversion;
pub const type_converter_convert_async = convertTypeAsync;
pub const type_converter_destroy_async = destroyTypeConverterAsync;
pub const conditionally_speculatable_attach_fallback_model = attachSpeculatabilityFallback;
pub const conditionally_speculatable_query_async = querySpeculatabilityAsync;
