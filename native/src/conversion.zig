const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const mlir_capi = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");
const diagnostic = @import("diagnostic.zig");

const LegalityDispatcher = kinda.callback_runtime.Dispatcher(.{"conversion_legality"});
const TypeDispatcher = kinda.callback_runtime.Dispatcher(.{
    "convert_type",
    "convert_types",
    "source_materialization",
    "target_materialization",
    "target_materialization_1_to_n",
});
const PatternDispatcher = kinda.callback_runtime.Dispatcher(.{
    "conversion_pattern",
    "conversion_pattern_1_to_n",
});

fn handleSlice(
    comptime Kind: type,
    environment: beam.env,
    count: isize,
    values: [*c]Kind.T,
) !beam.term {
    const len: usize = @intCast(count);
    if (len == 0) {
        const empty = [_]beam.term{};
        return beam.make_term_list(environment, &empty);
    }
    return kinda.callback_adapter.handleRange(Kind, environment, values[0..len]);
}

fn fetchHandleSlice(comptime Kind: type, environment: beam.env, list: beam.term) ![]Kind.T {
    const len = try beam.get_list_length(environment, list);
    const values = try std.heap.smp_allocator.alloc(Kind.T, len);
    errdefer std.heap.smp_allocator.free(values);

    var rest = list;
    for (values) |*value| {
        const head = try beam.get_head_and_iter(environment, &rest);
        value.* = try Kind.resource.fetch(environment, head);
    }
    return values;
}

fn HandleProjection(comptime Kind: type) type {
    return struct {
        values: []Kind.T,

        fn init(environment: beam.env, list: beam.term) !*@This() {
            const self = try std.heap.smp_allocator.create(@This());
            errdefer std.heap.smp_allocator.destroy(self);
            self.* = .{ .values = try fetchHandleSlice(Kind, environment, list) };
            return self;
        }

        fn deinit(self: *@This()) void {
            std.heap.smp_allocator.free(self.values);
            std.heap.smp_allocator.destroy(self);
        }
    };
}

fn replyHandleList(comptime Kind: type) type {
    return struct {
        fn nif(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const success = try beam.get_bool(environment, args[1]);
            const code = try beam.get_i64(environment, args[2]);
            var projection: usize = 0;
            var projected: ?*HandleProjection(Kind) = null;
            if (success and code == c.MlirTypeConverterConversionStatusSuccess) {
                projected = try HandleProjection(Kind).init(environment, args[3]);
                projection = @intFromPtr(projected.?);
            }

            const accepted = try kinda.callback_runtime.ReplyToken.complete(
                environment,
                args[0],
                .replied,
                success,
                code,
                projection,
            );
            if (!accepted) if (projected) |value| value.deinit();
            return beam.make_atom(environment, if (accepted) "ok" else "stale");
        }
    };
}

fn replyValue(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const success = try beam.get_bool(environment, args[1]);
    var projection: usize = 0;
    if (success and !beam.is_nil2(environment, args[2])) {
        const value = try mlir_capi.Value.resource.fetch(environment, args[2]);
        projection = @intFromPtr(value.ptr);
    }
    const accepted = try kinda.callback_runtime.ReplyToken.complete(
        environment,
        args[0],
        .replied,
        success,
        0,
        projection,
    );
    return beam.make_atom(environment, if (accepted) "ok" else "stale");
}

fn replyType(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
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

const LegalityState = struct {
    dispatcher: *LegalityDispatcher,

    fn deinit(self: *@This()) void {
        self.dispatcher.deinit();
        std.heap.smp_allocator.destroy(self);
    }

    fn callback(operation: mlir_capi.Operation.T, user_data: ?*anyopaque) callconv(.c) c.MlirConversionTargetLegality {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL));
        const environment = e.enif_alloc_env() orelse
            return c.MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL;
        const operation_term = kinda.callback_adapter.handle(mlir_capi.Operation, environment, operation) catch {
            e.enif_free_env(environment);
            return c.MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL;
        };
        const response = self.dispatcher.invoke(
            "conversion_legality",
            environment,
            .{operation_term},
        ) catch return c.MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL;
        return kinda.callback_adapter.scalarResult(
            c.MlirConversionTargetLegality,
            response,
            c.MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL,
        );
    }
};

const TargetRegistration = struct {
    target: mlir_capi.ConversionTarget.T,
    context: mlir_capi.Context.T,
    callbacks: std.array_list.Managed(*LegalityState),
    mutex: std.Io.Mutex = .init,
    closed: std.atomic.Value(bool) = .init(false),

    fn close(self: *@This()) bool {
        const io = std.Options.debug_io;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.closed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return false;
        c.mlirConversionTargetDestroy(self.target);
        for (self.callbacks.items) |callback| callback.deinit();
        self.callbacks.deinit();
        return true;
    }
};

fn destroyTargetRegistration(_: beam.env, object: ?*anyopaque) callconv(.c) void {
    const registration: *TargetRegistration = @ptrCast(@alignCast(object orelse return));
    _ = registration.close();
}

const TargetRegistrationResource = kinda.RawResourceType(
    TargetRegistration,
    "Beaver.MLIR.ConversionTarget.Registration",
    destroyTargetRegistration,
);

fn fetchTargetRegistration(environment: beam.env, term: beam.term) !*TargetRegistration {
    return TargetRegistrationResource.fetch(environment, term);
}

fn createTarget(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
    const registration = TargetRegistrationResource.alloc() catch
        return error.FailedToAllocateConversionTargetRegistration;
    registration.* = .{
        .target = c.mlirConversionTargetCreate(context),
        .context = context,
        .callbacks = std.array_list.Managed(*LegalityState).init(std.heap.smp_allocator),
    };
    const registration_term = e.enif_make_resource(environment, registration);
    TargetRegistrationResource.release(registration);
    var terms = [_]beam.term{
        beam.make_atom(environment, "managed_conversion_target"),
        try mlir_capi.ConversionTarget.resource.make_kind(environment, registration.target),
        registration_term,
    };
    return beam.make_tuple(environment, &terms);
}

const LegalityKind = enum { operation, dialect, recursively_legal, unknown };

fn addLegality(
    environment: beam.env,
    registration: *TargetRegistration,
    name_term: ?beam.term,
    callback_term: ?beam.term,
    timeout_term: beam.term,
    kind: LegalityKind,
) !beam.term {
    const io = std.Options.debug_io;
    registration.mutex.lockUncancelable(io);
    defer registration.mutex.unlock(io);
    if (registration.closed.load(.acquire)) return error.ConversionTargetClosed;

    const name = if (name_term) |term|
        try string_ref.get_binary_as_string_ref(environment, term)
    else
        mlir_capi.StringRef.T{ .data = null, .length = 0 };

    if (kind == .recursively_legal and callback_term != null and
        beam.is_nil2(environment, callback_term.?))
    {
        c.mlirConversionTargetMarkOpRecursivelyLegal(
            registration.target,
            name,
            null,
            null,
        );
        return beam.make_atom(environment, "ok");
    }

    const dispatcher = try LegalityDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try beam.get_u64(environment, timeout_term),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    dispatcher.setCallback("conversion_legality", callback_term.?);

    const state = try std.heap.smp_allocator.create(LegalityState);
    errdefer std.heap.smp_allocator.destroy(state);
    state.* = .{ .dispatcher = dispatcher };
    dispatcher_owned = false;
    errdefer state.deinit();
    try registration.callbacks.append(state);

    switch (kind) {
        .operation => c.mlirConversionTargetAddDynamicallyLegalOp(
            registration.target,
            name,
            LegalityState.callback,
            state,
        ),
        .dialect => c.mlirConversionTargetAddDynamicallyLegalDialect(
            registration.target,
            name,
            LegalityState.callback,
            state,
        ),
        .recursively_legal => c.mlirConversionTargetMarkOpRecursivelyLegal(
            registration.target,
            name,
            LegalityState.callback,
            state,
        ),
        .unknown => c.mlirConversionTargetMarkUnknownOpDynamicallyLegal(
            registration.target,
            LegalityState.callback,
            state,
        ),
    }
    return beam.make_atom(environment, "ok");
}

fn addDynamicOp(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addLegality(environment, try fetchTargetRegistration(environment, args[0]), args[1], args[2], args[3], .operation);
}

fn addDynamicDialect(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addLegality(environment, try fetchTargetRegistration(environment, args[0]), args[1], args[2], args[3], .dialect);
}

fn markRecursivelyLegal(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addLegality(environment, try fetchTargetRegistration(environment, args[0]), args[1], args[2], args[3], .recursively_legal);
}

fn markUnknownDynamic(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addLegality(environment, try fetchTargetRegistration(environment, args[0]), null, args[1], args[2], .unknown);
}

fn destroyTarget(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    _ = (try fetchTargetRegistration(environment, args[0])).close();
    return beam.make_atom(environment, "ok");
}

fn addStaticLegality(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const registration = try fetchTargetRegistration(environment, args[0]);
    const name = try string_ref.get_binary_as_string_ref(environment, args[1]);
    const kind = try beam.get_i64(environment, args[2]);
    const io = std.Options.debug_io;
    registration.mutex.lockUncancelable(io);
    defer registration.mutex.unlock(io);
    if (registration.closed.load(.acquire)) return error.ConversionTargetClosed;
    switch (kind) {
        0 => c.mlirConversionTargetAddLegalOp(registration.target, name),
        1 => c.mlirConversionTargetAddIllegalOp(registration.target, name),
        2 => c.mlirConversionTargetAddLegalDialect(registration.target, name),
        3 => c.mlirConversionTargetAddIllegalDialect(registration.target, name),
        else => return error.InvalidStaticLegalityKind,
    }
    return beam.make_atom(environment, "ok");
}

const TypeCallbackState = struct {
    dispatcher: *TypeDispatcher,

    fn deinit(self: *@This()) void {
        self.dispatcher.deinit();
        std.heap.smp_allocator.destroy(self);
    }

    fn conversion(
        type_: mlir_capi.Type.T,
        converted_type: [*c]mlir_capi.Type.T,
        user_data: ?*anyopaque,
    ) callconv(.c) c.MlirTypeConverterConversionStatus {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.MlirTypeConverterConversionStatusFailure));
        const environment = e.enif_alloc_env() orelse
            return c.MlirTypeConverterConversionStatusFailure;
        const type_term = kinda.callback_adapter.handle(mlir_capi.Type, environment, type_) catch {
            e.enif_free_env(environment);
            return c.MlirTypeConverterConversionStatusFailure;
        };
        const response = self.dispatcher.invoke("convert_type", environment, .{type_term}) catch
            return c.MlirTypeConverterConversionStatusFailure;
        const status = kinda.callback_adapter.scalarResult(
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

    fn conversion1ToN(
        type_: mlir_capi.Type.T,
        results: mlir_capi.TypeConverterConversionResults.T,
        user_data: ?*anyopaque,
    ) callconv(.c) c.MlirTypeConverterConversionStatus {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.MlirTypeConverterConversionStatusFailure));
        const environment = e.enif_alloc_env() orelse
            return c.MlirTypeConverterConversionStatusFailure;
        const type_term = kinda.callback_adapter.handle(mlir_capi.Type, environment, type_) catch {
            e.enif_free_env(environment);
            return c.MlirTypeConverterConversionStatusFailure;
        };
        const response = self.dispatcher.invoke("convert_types", environment, .{type_term}) catch
            return c.MlirTypeConverterConversionStatusFailure;
        const status = kinda.callback_adapter.scalarResult(
            c.MlirTypeConverterConversionStatus,
            response,
            c.MlirTypeConverterConversionStatusFailure,
        );
        if (status == c.MlirTypeConverterConversionStatusSuccess) {
            const projection = kinda.callback_adapter.projection(response) orelse
                return c.MlirTypeConverterConversionStatusFailure;
            const projected: *HandleProjection(mlir_capi.Type) = @ptrFromInt(projection);
            defer projected.deinit();
            for (projected.values) |value| {
                c.mlirTypeConverterConversionResultsAppend(results, value);
            }
        }
        return status;
    }

    fn sourceMaterialization(
        rewriter: mlir_capi.RewriterBase.T,
        output_type: mlir_capi.Type.T,
        n_inputs: isize,
        inputs: [*c]mlir_capi.Value.T,
        location: mlir_capi.Location.T,
        user_data: ?*anyopaque,
    ) callconv(.c) mlir_capi.Value.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return .{ .ptr = null }));
        const environment = e.enif_alloc_env() orelse return .{ .ptr = null };
        const args = .{
            kinda.callback_adapter.handle(mlir_capi.RewriterBase, environment, rewriter) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
            kinda.callback_adapter.handle(mlir_capi.Type, environment, output_type) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
            handleSlice(mlir_capi.Value, environment, n_inputs, inputs) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
            kinda.callback_adapter.handle(mlir_capi.Location, environment, location) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
        };
        const response = self.dispatcher.invoke("source_materialization", environment, args) catch
            return .{ .ptr = null };
        const projection = kinda.callback_adapter.projection(response) orelse return .{ .ptr = null };
        return .{ .ptr = @ptrFromInt(projection) };
    }

    fn targetMaterialization(
        rewriter: mlir_capi.RewriterBase.T,
        output_type: mlir_capi.Type.T,
        n_inputs: isize,
        inputs: [*c]mlir_capi.Value.T,
        location: mlir_capi.Location.T,
        original_type: mlir_capi.Type.T,
        user_data: ?*anyopaque,
    ) callconv(.c) mlir_capi.Value.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return .{ .ptr = null }));
        const environment = e.enif_alloc_env() orelse return .{ .ptr = null };
        const args = .{
            kinda.callback_adapter.handle(mlir_capi.RewriterBase, environment, rewriter) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
            kinda.callback_adapter.handle(mlir_capi.Type, environment, output_type) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
            handleSlice(mlir_capi.Value, environment, n_inputs, inputs) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
            kinda.callback_adapter.handle(mlir_capi.Location, environment, location) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
            kinda.callback_adapter.handle(mlir_capi.Type, environment, original_type) catch {
                e.enif_free_env(environment);
                return .{ .ptr = null };
            },
        };
        const response = self.dispatcher.invoke("target_materialization", environment, args) catch
            return .{ .ptr = null };
        const projection = kinda.callback_adapter.projection(response) orelse return .{ .ptr = null };
        return .{ .ptr = @ptrFromInt(projection) };
    }

    fn targetMaterialization1ToN(
        rewriter: mlir_capi.RewriterBase.T,
        n_output_types: isize,
        output_types: [*c]mlir_capi.Type.T,
        n_inputs: isize,
        inputs: [*c]mlir_capi.Value.T,
        location: mlir_capi.Location.T,
        original_type: mlir_capi.Type.T,
        outputs: [*c]mlir_capi.Value.T,
        user_data: ?*anyopaque,
    ) callconv(.c) mlir_capi.LogicalResult.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.beaverLogicalResultFailure()));
        const environment = e.enif_alloc_env() orelse return c.beaverLogicalResultFailure();
        const original_term = if (c.mlirTypeIsNull(original_type))
            beam.make_nil(environment)
        else
            kinda.callback_adapter.handle(mlir_capi.Type, environment, original_type) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            };
        const args = .{
            kinda.callback_adapter.handle(mlir_capi.RewriterBase, environment, rewriter) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
            handleSlice(mlir_capi.Type, environment, n_output_types, output_types) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
            handleSlice(mlir_capi.Value, environment, n_inputs, inputs) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
            kinda.callback_adapter.handle(mlir_capi.Location, environment, location) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
            original_term,
        };
        const response = self.dispatcher.invoke("target_materialization_1_to_n", environment, args) catch
            return c.beaverLogicalResultFailure();
        if (!response.success) return c.beaverLogicalResultFailure();
        const projection = kinda.callback_adapter.projection(response) orelse
            return c.beaverLogicalResultFailure();
        const projected: *HandleProjection(mlir_capi.Value) = @ptrFromInt(projection);
        defer projected.deinit();
        if (projected.values.len != @as(usize, @intCast(n_output_types)))
            return c.beaverLogicalResultFailure();
        if (projected.values.len != 0)
            @memcpy(outputs[0..projected.values.len], projected.values);
        return c.beaverLogicalResultSuccess();
    }
};

const TypeConverterRegistration = struct {
    converter: mlir_capi.TypeConverter.T,
    callbacks: std.array_list.Managed(*TypeCallbackState),
    mutex: std.Io.Mutex = .init,
    closed: std.atomic.Value(bool) = .init(false),
    pattern_users: std.atomic.Value(usize) = .init(0),

    fn closeLocked(self: *@This()) bool {
        if (self.closed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return false;
        c.mlirTypeConverterDestroy(self.converter);
        for (self.callbacks.items) |callback| callback.deinit();
        self.callbacks.deinit();
        return true;
    }

    fn close(self: *@This()) bool {
        const io = std.Options.debug_io;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.closeLocked();
    }
};

fn destroyTypeConverterRegistration(_: beam.env, object: ?*anyopaque) callconv(.c) void {
    const registration: *TypeConverterRegistration = @ptrCast(@alignCast(object orelse return));
    _ = registration.close();
}

const TypeConverterRegistrationResource = kinda.RawResourceType(
    TypeConverterRegistration,
    "Beaver.MLIR.TypeConverter.Registration",
    destroyTypeConverterRegistration,
);

fn fetchTypeConverterRegistration(environment: beam.env, term: beam.term) !*TypeConverterRegistration {
    return TypeConverterRegistrationResource.fetch(environment, term);
}

fn createTypeConverter(environment: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    const registration = TypeConverterRegistrationResource.alloc() catch
        return error.FailedToAllocateTypeConverterRegistration;
    registration.* = .{
        .converter = c.mlirTypeConverterCreate(),
        .callbacks = std.array_list.Managed(*TypeCallbackState).init(std.heap.smp_allocator),
    };
    const registration_term = e.enif_make_resource(environment, registration);
    TypeConverterRegistrationResource.release(registration);
    var terms = [_]beam.term{
        beam.make_atom(environment, "managed_type_converter"),
        try mlir_capi.TypeConverter.resource.make_kind(environment, registration.converter),
        registration_term,
    };
    return beam.make_tuple(environment, &terms);
}

const TypeCallbackKind = enum { conversion, conversion_1_to_n, source, target, target_1_to_n };

fn addTypeCallback(
    environment: beam.env,
    registration: *TypeConverterRegistration,
    callback: beam.term,
    timeout: beam.term,
    kind: TypeCallbackKind,
) !beam.term {
    const io = std.Options.debug_io;
    registration.mutex.lockUncancelable(io);
    defer registration.mutex.unlock(io);
    if (registration.closed.load(.acquire)) return error.TypeConverterClosed;

    const dispatcher = try TypeDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try beam.get_u64(environment, timeout),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    switch (kind) {
        .conversion => dispatcher.setCallback("convert_type", callback),
        .conversion_1_to_n => dispatcher.setCallback("convert_types", callback),
        .source => dispatcher.setCallback("source_materialization", callback),
        .target => dispatcher.setCallback("target_materialization", callback),
        .target_1_to_n => dispatcher.setCallback("target_materialization_1_to_n", callback),
    }
    const state = try std.heap.smp_allocator.create(TypeCallbackState);
    errdefer std.heap.smp_allocator.destroy(state);
    state.* = .{ .dispatcher = dispatcher };
    dispatcher_owned = false;
    errdefer state.deinit();
    try registration.callbacks.append(state);

    switch (kind) {
        .conversion => c.mlirTypeConverterAddConversion(registration.converter, TypeCallbackState.conversion, state),
        .conversion_1_to_n => c.mlirTypeConverterAdd1ToNConversion(registration.converter, TypeCallbackState.conversion1ToN, state),
        .source => c.mlirTypeConverterAddSourceMaterialization(registration.converter, TypeCallbackState.sourceMaterialization, state),
        .target => c.mlirTypeConverterAddTargetMaterialization(registration.converter, TypeCallbackState.targetMaterialization, state),
        .target_1_to_n => c.mlirTypeConverterAdd1ToNTargetMaterialization(registration.converter, TypeCallbackState.targetMaterialization1ToN, state),
    }
    return beam.make_atom(environment, "ok");
}

fn addTypeConversion(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addTypeCallback(environment, try fetchTypeConverterRegistration(environment, args[0]), args[1], args[2], .conversion);
}

fn addTypeConversion1ToN(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addTypeCallback(environment, try fetchTypeConverterRegistration(environment, args[0]), args[1], args[2], .conversion_1_to_n);
}

fn addSourceMaterialization(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addTypeCallback(environment, try fetchTypeConverterRegistration(environment, args[0]), args[1], args[2], .source);
}

fn addTargetMaterialization(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addTypeCallback(environment, try fetchTypeConverterRegistration(environment, args[0]), args[1], args[2], .target);
}

fn addTargetMaterialization1ToN(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return addTypeCallback(environment, try fetchTypeConverterRegistration(environment, args[0]), args[1], args[2], .target_1_to_n);
}

const TypeConverterWorker = struct {
    registration: *TypeConverterRegistration,
    recipient: beam.pid,
    environment: beam.env,
    id: beam.term,
    type_: mlir_capi.Type.T,

    fn run(self: *@This()) void {
        const registration = self.registration;
        const owned_environment = self.environment;
        defer std.heap.smp_allocator.destroy(self);
        const environment = e.enif_alloc_env() orelse {
            e.enif_free_env(owned_environment);
            e.enif_release_resource(registration);
            std.heap.smp_allocator.destroy(self);
            return;
        };
        defer e.enif_free_env(environment);

        const io = std.Options.debug_io;
        self.registration.mutex.lockUncancelable(io);
        const converted = if (self.registration.closed.load(.acquire))
            mlir_capi.Type.T{ .ptr = null }
        else
            c.mlirTypeConverterConvertType(self.registration.converter, self.type_);
        self.registration.mutex.unlock(io);
        // Complete native cleanup before notifying the BEAM: once
        // `type_converter_done` is delivered the receiver may tear down the
        // context, so nothing may reference it afterwards.
        const id_term = e.enif_make_copy(environment, self.id);
        e.enif_free_env(owned_environment);
        e.enif_release_resource(registration);
        var terms = [_]beam.term{
            beam.make_atom(environment, "type_converter_done"),
            id_term,
            if (c.mlirTypeIsNull(converted))
                beam.make_nil(environment)
            else
                mlir_capi.Type.resource.make_kind(environment, converted) catch beam.make_nil(environment),
        };
        _ = beam.send_advanced(null, self.recipient, environment, beam.make_tuple(environment, &terms));
    }
};

fn convertTypeAsync(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const registration = try fetchTypeConverterRegistration(environment, args[0]);
    const worker = try std.heap.smp_allocator.create(TypeConverterWorker);
    errdefer std.heap.smp_allocator.destroy(worker);
    const owned_env = e.enif_alloc_env() orelse return error.FailedToAllocateEnvironment;
    errdefer e.enif_free_env(owned_env);
    const id = e.enif_make_ref(environment);
    worker.* = .{
        .registration = registration,
        .recipient = try beam.self(environment),
        .environment = owned_env,
        .id = e.enif_make_copy(owned_env, id),
        .type_ = try mlir_capi.Type.resource.fetch(environment, args[1]),
    };
    e.enif_keep_resource(registration);
    errdefer e.enif_release_resource(registration);
    const thread = try std.Thread.spawn(.{}, TypeConverterWorker.run, .{worker});
    thread.detach();
    return id;
}

fn destroyTypeConverter(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const registration = try fetchTypeConverterRegistration(environment, args[0]);
    const io = std.Options.debug_io;
    registration.mutex.lockUncancelable(io);
    defer registration.mutex.unlock(io);
    if (registration.pattern_users.load(.acquire) != 0) return error.TypeConverterInUseByPatterns;
    _ = registration.closeLocked();
    return beam.make_atom(environment, "ok");
}

const ConversionPatternState = struct {
    dispatcher: *PatternDispatcher,
    converter_registration: *TypeConverterRegistration,

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        self.dispatcher.deinit();
        _ = self.converter_registration.pattern_users.fetchSub(1, .acq_rel);
        e.enif_release_resource(self.converter_registration);
        std.heap.smp_allocator.destroy(self);
    }

    fn rewrite(
        _: mlir_capi.ConversionPattern.T,
        operation: mlir_capi.Operation.T,
        n_operands: isize,
        operands: [*c]mlir_capi.Value.T,
        rewriter: mlir_capi.ConversionPatternRewriter.T,
        user_data: ?*anyopaque,
    ) callconv(.c) mlir_capi.LogicalResult.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.beaverLogicalResultFailure()));
        const environment = e.enif_alloc_env() orelse return c.beaverLogicalResultFailure();
        const args = .{
            kinda.callback_adapter.handle(mlir_capi.Operation, environment, operation) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
            handleSlice(mlir_capi.Value, environment, n_operands, operands) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
            kinda.callback_adapter.handle(mlir_capi.ConversionPatternRewriter, environment, rewriter) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
        };
        const response = self.dispatcher.invoke("conversion_pattern", environment, args) catch
            return c.beaverLogicalResultFailure();
        return if (response.success) c.beaverLogicalResultSuccess() else c.beaverLogicalResultFailure();
    }

    fn rewrite1ToN(
        _: mlir_capi.ConversionPattern.T,
        operation: mlir_capi.Operation.T,
        n_ranges: isize,
        range_sizes: [*c]isize,
        n_operands: isize,
        operands: [*c]mlir_capi.Value.T,
        rewriter: mlir_capi.ConversionPatternRewriter.T,
        user_data: ?*anyopaque,
    ) callconv(.c) mlir_capi.LogicalResult.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.beaverLogicalResultFailure()));
        const environment = e.enif_alloc_env() orelse return c.beaverLogicalResultFailure();
        const count: usize = @intCast(n_ranges);
        const ranges = beam.allocator.alloc(beam.term, count) catch {
            e.enif_free_env(environment);
            return c.beaverLogicalResultFailure();
        };
        defer beam.allocator.free(ranges);
        var offset: usize = 0;
        for (0..count) |index| {
            const len: usize = @intCast(range_sizes[index]);
            if (offset + len > @as(usize, @intCast(n_operands))) {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            }
            if (len == 0) {
                const empty = [_]beam.term{};
                ranges[index] = beam.make_term_list(environment, &empty);
            } else {
                ranges[index] = kinda.callback_adapter.handleRange(
                    mlir_capi.Value,
                    environment,
                    operands[offset .. offset + len],
                ) catch {
                    e.enif_free_env(environment);
                    return c.beaverLogicalResultFailure();
                };
            }
            offset += len;
        }
        const args = .{
            kinda.callback_adapter.handle(mlir_capi.Operation, environment, operation) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
            beam.make_term_list(environment, ranges),
            kinda.callback_adapter.handle(mlir_capi.ConversionPatternRewriter, environment, rewriter) catch {
                e.enif_free_env(environment);
                return c.beaverLogicalResultFailure();
            },
        };
        const response = self.dispatcher.invoke("conversion_pattern_1_to_n", environment, args) catch
            return c.beaverLogicalResultFailure();
        return if (response.success) c.beaverLogicalResultSuccess() else c.beaverLogicalResultFailure();
    }
};

fn addConversionPattern(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const set = try mlir_capi.RewritePatternSet.resource.fetch(environment, args[0]);
    const root_name = try string_ref.get_binary_as_string_ref(environment, args[1]);
    const benefit = try beam.get(c_uint, environment, args[2]);
    const context = try mlir_capi.Context.resource.fetch(environment, args[3]);
    const registration = try fetchTypeConverterRegistration(environment, args[4]);
    const io = std.Options.debug_io;
    registration.mutex.lockUncancelable(io);
    if (registration.closed.load(.acquire)) {
        registration.mutex.unlock(io);
        return error.TypeConverterClosed;
    }
    _ = registration.pattern_users.fetchAdd(1, .acq_rel);
    registration.mutex.unlock(io);
    var pattern_user_owned = true;
    errdefer {
        if (pattern_user_owned) _ = registration.pattern_users.fetchSub(1, .acq_rel);
    }
    const dispatcher = try PatternDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try beam.get_u64(environment, args[7]),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    const one_to_n = try beam.get_bool(environment, args[6]);
    if (one_to_n)
        dispatcher.setCallback("conversion_pattern_1_to_n", args[5])
    else
        dispatcher.setCallback("conversion_pattern", args[5]);

    const state = try std.heap.smp_allocator.create(ConversionPatternState);
    errdefer std.heap.smp_allocator.destroy(state);
    state.* = .{ .dispatcher = dispatcher, .converter_registration = registration };
    dispatcher_owned = false;
    e.enif_keep_resource(registration);
    errdefer e.enif_release_resource(registration);

    const callbacks = c.MlirConversionPatternCallbacks{
        .construct = null,
        .destruct = ConversionPatternState.destruct,
        .matchAndRewrite = if (one_to_n) null else ConversionPatternState.rewrite,
        .matchAndRewrite1ToN = if (one_to_n) ConversionPatternState.rewrite1ToN else null,
    };
    const pattern = c.mlirOpConversionPatternCreate(
        root_name,
        benefit,
        context,
        registration.converter,
        callbacks,
        state,
        0,
        null,
    );
    c.mlirRewritePatternSetAdd(set, c.mlirConversionPatternAsRewritePattern(pattern));
    pattern_user_owned = false;
    return beam.make_atom(environment, "ok");
}

const ConversionWorker = struct {
    target: *TargetRegistration,
    operation: mlir_capi.Operation.T,
    patterns: mlir_capi.FrozenRewritePatternSet.T,
    config: mlir_capi.ConversionConfig.T,
    owns_patterns: bool,
    full: bool,
    recipient: beam.pid,
    environment: beam.env,
    id: beam.term,

    fn apply(environment: beam.env, args: anytype) !beam.term {
        const logical_result = if (args.full)
            c.mlirApplyFullConversion(args.operation, args.target, args.patterns, args.config)
        else
            c.mlirApplyPartialConversion(args.operation, args.target, args.patterns, args.config);
        return mlir_capi.LogicalResult.resource.make_kind(environment, logical_result);
    }

    fn run(worker: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(worker orelse return));
        const target = self.target;
        const owned_environment = self.environment;
        const config = self.config;
        const patterns = self.patterns;
        const owns_patterns = self.owns_patterns;
        var config_owned = true;
        var patterns_owned = owns_patterns;
        defer std.heap.smp_allocator.destroy(self);
        const environment = e.enif_alloc_env() orelse {
            if (patterns_owned) c.mlirFrozenRewritePatternSetDestroy(patterns);
            if (config_owned) c.mlirConversionConfigDestroy(config);
            e.enif_free_env(owned_environment);
            e.enif_release_resource(target);
            std.heap.smp_allocator.destroy(self);
            return;
        };
        defer e.enif_free_env(environment);

        const io = std.Options.debug_io;
        self.target.mutex.lockUncancelable(io);
        const result = if (self.target.closed.load(.acquire))
            null
        else
            diagnostic.call_with_diagnostics(
                environment,
                self.target.context,
                apply,
                .{ environment, .{
                    .full = self.full,
                    .operation = self.operation,
                    .target = self.target.target,
                    .patterns = self.patterns,
                    .config = self.config,
                } },
            ) catch null;
        self.target.mutex.unlock(io);

        if (patterns_owned) {
            c.mlirFrozenRewritePatternSetDestroy(patterns);
            patterns_owned = false;
        }
        if (config_owned) {
            c.mlirConversionConfigDestroy(config);
            config_owned = false;
        }
        // Complete native cleanup before notifying the BEAM: once
        // `conversion_done` is delivered the receiver may destroy the MLIR
        // context, so nothing may reference it afterwards.
        const id_term = e.enif_make_copy(environment, self.id);
        e.enif_free_env(owned_environment);
        e.enif_release_resource(target);

        var terms = [_]beam.term{
            beam.make_atom(environment, "conversion_done"),
            id_term,
            result orelse beam.make_nil(environment),
        };
        _ = beam.send_advanced(null, self.recipient, environment, beam.make_tuple(environment, &terms));
    }
};

fn applyConversionAsync(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const target = try fetchTargetRegistration(environment, args[1]);
    const patterns = try mlir_capi.FrozenRewritePatternSet.resource.fetch(environment, args[3]);
    const owns_patterns = try beam.get_bool(environment, args[4]);
    const folding_mode = try beam.get_i64(environment, args[5]);
    const materializations = try beam.get_i64(environment, args[6]);
    const config = c.mlirConversionConfigCreate();
    var config_owned = true;
    errdefer if (config_owned) c.mlirConversionConfigDestroy(config);
    if (folding_mode >= 0)
        c.mlirConversionConfigSetFoldingMode(config, @intCast(folding_mode));
    if (materializations >= 0)
        c.mlirConversionConfigEnableBuildMaterializations(config, materializations != 0);

    const worker = try std.heap.smp_allocator.create(ConversionWorker);
    errdefer std.heap.smp_allocator.destroy(worker);
    const owned_env = e.enif_alloc_env() orelse return error.FailedToAllocateEnvironment;
    errdefer e.enif_free_env(owned_env);
    const id = e.enif_make_ref(environment);
    worker.* = .{
        .target = target,
        .operation = try mlir_capi.Operation.resource.fetch(environment, args[2]),
        .patterns = patterns,
        .config = config,
        .owns_patterns = owns_patterns,
        .full = try beam.get_bool(environment, args[0]),
        .recipient = try beam.self(environment),
        .environment = owned_env,
        .id = e.enif_make_copy(owned_env, id),
    };
    e.enif_keep_resource(target);
    errdefer e.enif_release_resource(target);
    if (!c.beaverContextAddWork(target.context, ConversionWorker.run, worker))
        return error.ContextMultithreadingDisabled;
    config_owned = false;
    return id;
}

pub fn open(environment: beam.env) void {
    TargetRegistrationResource.open(environment);
    TypeConverterRegistrationResource.open(environment);
}

pub const nifs = .{
    prelude.beaverRawNIF(@This(), "conversion_target_create", 1),
    prelude.beaverRawNIF(@This(), "conversion_target_add_static", 3),
    prelude.beaverRawNIF(@This(), "conversion_target_add_dynamic_op", 4),
    prelude.beaverRawNIF(@This(), "conversion_target_add_dynamic_dialect", 4),
    prelude.beaverRawNIF(@This(), "conversion_target_mark_recursively_legal", 4),
    prelude.beaverRawNIF(@This(), "conversion_target_mark_unknown_dynamic", 3),
    prelude.beaverRawNIF(@This(), "conversion_target_destroy", 1),
    prelude.beaverRawNIF(@This(), "type_converter_create", 0),
    prelude.beaverRawNIF(@This(), "type_converter_add_conversion", 3),
    prelude.beaverRawNIF(@This(), "type_converter_add_1_to_n_conversion", 3),
    prelude.beaverRawNIF(@This(), "type_converter_add_source_materialization", 3),
    prelude.beaverRawNIF(@This(), "type_converter_add_target_materialization", 3),
    prelude.beaverRawNIF(@This(), "type_converter_add_1_to_n_target_materialization", 3),
    prelude.beaverRawNIF(@This(), "type_converter_convert_async", 2),
    prelude.beaverRawNIF(@This(), "type_converter_destroy", 1),
    prelude.beaverRawNIF(@This(), "type_converter_reply_callback", 4),
    prelude.beaverRawNIF(@This(), "type_converter_reply_types", 4),
    prelude.beaverRawNIF(@This(), "type_converter_reply_value", 3),
    prelude.beaverRawNIF(@This(), "type_converter_reply_values", 4),
    prelude.beaverRawNIF(@This(), "conversion_pattern_add", 8),
    prelude.beaverRawNIF(@This(), "apply_conversion_async", 7),
};

pub const conversion_target_create = createTarget;
pub const conversion_target_add_static = addStaticLegality;
pub const conversion_target_add_dynamic_op = addDynamicOp;
pub const conversion_target_add_dynamic_dialect = addDynamicDialect;
pub const conversion_target_mark_recursively_legal = markRecursivelyLegal;
pub const conversion_target_mark_unknown_dynamic = markUnknownDynamic;
pub const conversion_target_destroy = destroyTarget;
pub const type_converter_create = createTypeConverter;
pub const type_converter_add_conversion = addTypeConversion;
pub const type_converter_add_1_to_n_conversion = addTypeConversion1ToN;
pub const type_converter_add_source_materialization = addSourceMaterialization;
pub const type_converter_add_target_materialization = addTargetMaterialization;
pub const type_converter_add_1_to_n_target_materialization = addTargetMaterialization1ToN;
pub const type_converter_convert_async = convertTypeAsync;
pub const type_converter_destroy = destroyTypeConverter;
pub const type_converter_reply_callback = replyType;
pub const type_converter_reply_types = replyHandleList(mlir_capi.Type).nif;
pub const type_converter_reply_value = replyValue;
pub const type_converter_reply_values = replyHandleList(mlir_capi.Value).nif;
pub const conversion_pattern_add = addConversionPattern;
pub const apply_conversion_async = applyConversionAsync;
