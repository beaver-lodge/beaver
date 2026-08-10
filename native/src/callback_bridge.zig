const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const mlir_capi = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");
const diagnostic = @import("diagnostic.zig");

const SpeculatabilityDispatcher = kinda.callback_runtime.Dispatcher(.{"get_speculatability"});
const MemoryEffectsDispatcher = kinda.callback_runtime.Dispatcher(.{"get_effects"});
const TransformDispatcher = kinda.callback_runtime.Dispatcher(.{ "apply", "allows_repeated_handle_operands" });
const PatternDescriptorDispatcher = kinda.callback_runtime.Dispatcher(.{ "populate_patterns", "populate_patterns_with_state" });
const DynamicTraitDispatcher = kinda.callback_runtime.Dispatcher(.{ "verify", "verify_regions" });

var speculatability_state_type: beam.resource_type = undefined;
var memory_effects_state_type: beam.resource_type = undefined;
var transform_state_type: beam.resource_type = undefined;
var pattern_descriptor_state_type: beam.resource_type = undefined;
var dynamic_trait_state_type: beam.resource_type = undefined;

fn notifyReleased(recipient: beam.pid) void {
    const environment = e.enif_alloc_env() orelse return;
    defer e.enif_free_env(environment);
    _ = beam.send_advanced(
        null,
        recipient,
        environment,
        beam.make_atom(environment, "external_interface_released"),
    );
}

fn makeHandle(
    comptime Kind: type,
    environment: beam.env,
    value: Kind.T,
) ?beam.term {
    return kinda.callback_adapter.handle(Kind, environment, value) catch {
        e.enif_free_env(environment);
        return null;
    };
}

fn releaseModel(comptime State: type, self: *State) void {
    if (self.model_released.cmpxchgStrong(false, true, .acq_rel, .acquire) == null)
        e.enif_release_resource(self);
}

fn attachmentTerm(
    comptime State: type,
    environment: beam.env,
    dispatcher_id: beam.term,
    state: *State,
) beam.term {
    // Keep the reference returned by enif_alloc_resource for the MLIR model.
    // enif_make_resource adds an independent BEAM-term reference. If the
    // attachment process dies and drops that term, the model reference keeps
    // both this state and its dispatcher alive until the model destructor calls
    // releaseModel. Do not release the allocation reference here.
    var terms = [_]beam.term{
        dispatcher_id,
        e.enif_make_resource(environment, state),
    };
    return beam.make_tuple(environment, &terms);
}

const SpeculatabilityState = struct {
    dispatcher: *SpeculatabilityDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn construct(_: ?*anyopaque) callconv(.c) void {}

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        notifyReleased(self.dispatcher.handler);
        releaseModel(@This(), self);
    }

    fn getSpeculatability(
        operation: mlir_capi.Operation.T,
        user_data: ?*anyopaque,
    ) callconv(.c) c.MlirSpeculatability {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.MlirSpeculatabilityNotSpeculatable));
        const environment = e.enif_alloc_env() orelse
            return c.MlirSpeculatabilityNotSpeculatable;
        const operation_term = makeHandle(mlir_capi.Operation, environment, operation) orelse
            return c.MlirSpeculatabilityNotSpeculatable;
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

const MemoryEffectsState = struct {
    dispatcher: *MemoryEffectsDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn construct(_: ?*anyopaque) callconv(.c) void {}

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        notifyReleased(self.dispatcher.handler);
        releaseModel(@This(), self);
    }

    fn appendConservativeEffect(effects: mlir_capi.MemoryEffectInstancesList.T) void {
        const instance = c.mlirMemoryEffectInstanceCreate(
            c.mlirMemoryEffectsWriteGet(),
            c.mlirAttributeGetNull(),
            0,
            false,
            c.mlirSideEffectsDefaultResourceGet(),
        );
        defer c.mlirMemoryEffectInstanceDestroy(instance);
        c.mlirMemoryEffectInstancesListAppend(effects, instance);
    }

    fn getEffects(
        operation: mlir_capi.Operation.T,
        effects: mlir_capi.MemoryEffectInstancesList.T,
        user_data: ?*anyopaque,
    ) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse {
            appendConservativeEffect(effects);
            return;
        }));
        const environment = e.enif_alloc_env() orelse {
            appendConservativeEffect(effects);
            return;
        };
        const operation_term = makeHandle(mlir_capi.Operation, environment, operation) orelse {
            appendConservativeEffect(effects);
            return;
        };
        const effects_term = kinda.callback_adapter.handle(
            mlir_capi.MemoryEffectInstancesList,
            environment,
            effects,
        ) catch {
            e.enif_free_env(environment);
            appendConservativeEffect(effects);
            return;
        };
        const response = self.dispatcher.invoke(
            "get_effects",
            environment,
            .{ operation_term, effects_term },
        ) catch {
            appendConservativeEffect(effects);
            return;
        };
        if (!response.success) appendConservativeEffect(effects);
    }
};

const TransformState = struct {
    dispatcher: *TransformDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn construct(_: ?*anyopaque) callconv(.c) void {}

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        notifyReleased(self.dispatcher.handler);
        releaseModel(@This(), self);
    }

    fn apply(
        operation: mlir_capi.Operation.T,
        rewriter: mlir_capi.TransformRewriter.T,
        results: mlir_capi.TransformResults.T,
        state: mlir_capi.TransformState.T,
        user_data: ?*anyopaque,
    ) callconv(.c) c.MlirDiagnosedSilenceableFailure {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.MlirDiagnosedSilenceableFailureDefiniteFailure));
        const environment = e.enif_alloc_env() orelse
            return c.MlirDiagnosedSilenceableFailureDefiniteFailure;
        const operation_term = makeHandle(mlir_capi.Operation, environment, operation) orelse
            return c.MlirDiagnosedSilenceableFailureDefiniteFailure;
        const rewriter_term = kinda.callback_adapter.handle(
            mlir_capi.TransformRewriter,
            environment,
            rewriter,
        ) catch {
            e.enif_free_env(environment);
            return c.MlirDiagnosedSilenceableFailureDefiniteFailure;
        };
        const results_term = kinda.callback_adapter.handle(
            mlir_capi.TransformResults,
            environment,
            results,
        ) catch {
            e.enif_free_env(environment);
            return c.MlirDiagnosedSilenceableFailureDefiniteFailure;
        };
        const state_term = kinda.callback_adapter.handle(
            mlir_capi.TransformState,
            environment,
            state,
        ) catch {
            e.enif_free_env(environment);
            return c.MlirDiagnosedSilenceableFailureDefiniteFailure;
        };
        const response = self.dispatcher.invoke(
            "apply",
            environment,
            .{ operation_term, rewriter_term, results_term, state_term },
        ) catch return c.MlirDiagnosedSilenceableFailureDefiniteFailure;
        return kinda.callback_adapter.enumResult(
            c.MlirDiagnosedSilenceableFailure,
            response,
            c.MlirDiagnosedSilenceableFailureDefiniteFailure,
        );
    }

    fn allowsRepeatedHandleOperands(
        operation: mlir_capi.Operation.T,
        user_data: ?*anyopaque,
    ) callconv(.c) bool {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return false));
        if (!self.dispatcher.hasCallback("allows_repeated_handle_operands")) return false;
        const environment = e.enif_alloc_env() orelse return false;
        const operation_term = makeHandle(mlir_capi.Operation, environment, operation) orelse
            return false;
        const response = self.dispatcher.invoke(
            "allows_repeated_handle_operands",
            environment,
            .{operation_term},
        ) catch return false;
        return kinda.callback_adapter.scalarResult(u1, response, 0) != 0;
    }
};

const PatternDescriptorState = struct {
    dispatcher: *PatternDescriptorDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn construct(_: ?*anyopaque) callconv(.c) void {}

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        notifyReleased(self.dispatcher.handler);
        releaseModel(@This(), self);
    }

    fn populatePatterns(
        operation: mlir_capi.Operation.T,
        patterns: mlir_capi.RewritePatternSet.T,
        user_data: ?*anyopaque,
    ) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        const environment = e.enif_alloc_env() orelse return;
        const operation_term = makeHandle(mlir_capi.Operation, environment, operation) orelse return;
        const patterns_term = kinda.callback_adapter.handle(
            mlir_capi.RewritePatternSet,
            environment,
            patterns,
        ) catch {
            e.enif_free_env(environment);
            return;
        };
        _ = self.dispatcher.invoke(
            "populate_patterns",
            environment,
            .{ operation_term, patterns_term },
        ) catch {};
    }

    fn populatePatternsWithState(
        operation: mlir_capi.Operation.T,
        patterns: mlir_capi.RewritePatternSet.T,
        state: mlir_capi.TransformState.T,
        user_data: ?*anyopaque,
    ) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        const environment = e.enif_alloc_env() orelse return;
        const operation_term = makeHandle(mlir_capi.Operation, environment, operation) orelse return;
        const patterns_term = kinda.callback_adapter.handle(
            mlir_capi.RewritePatternSet,
            environment,
            patterns,
        ) catch {
            e.enif_free_env(environment);
            return;
        };
        const state_term = kinda.callback_adapter.handle(
            mlir_capi.TransformState,
            environment,
            state,
        ) catch {
            e.enif_free_env(environment);
            return;
        };
        _ = self.dispatcher.invoke(
            "populate_patterns_with_state",
            environment,
            .{ operation_term, patterns_term, state_term },
        ) catch {};
    }
};

const DynamicTraitState = struct {
    dispatcher: *DynamicTraitDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn construct(_: ?*anyopaque) callconv(.c) void {}

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        notifyReleased(self.dispatcher.handler);
        releaseModel(@This(), self);
    }

    fn invokeVerifier(
        self: *@This(),
        comptime callback: []const u8,
        operation: mlir_capi.Operation.T,
    ) mlir_capi.LogicalResult.T {
        if (!self.dispatcher.hasCallback(callback)) return c.beaverLogicalResultSuccess();
        const environment = e.enif_alloc_env() orelse
            return callbackFailure(operation);
        const operation_term = makeHandle(mlir_capi.Operation, environment, operation) orelse
            return callbackFailure(operation);
        const response = self.dispatcher.invoke(callback, environment, .{operation_term}) catch
            return callbackFailure(operation);
        if (response.status != .replied) return callbackFailure(operation);
        return if (response.success) c.beaverLogicalResultSuccess() else c.beaverLogicalResultFailure();
    }

    fn callbackFailure(operation: mlir_capi.Operation.T) mlir_capi.LogicalResult.T {
        c.mlirEmitError(
            c.mlirOperationGetLocation(operation),
            "dynamic trait callback timed out or its owner is unavailable",
        );
        return c.beaverLogicalResultFailure();
    }

    fn verify(
        operation: mlir_capi.Operation.T,
        user_data: ?*anyopaque,
    ) callconv(.c) mlir_capi.LogicalResult.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.beaverLogicalResultFailure()));
        return self.invokeVerifier("verify", operation);
    }

    fn verifyRegions(
        operation: mlir_capi.Operation.T,
        user_data: ?*anyopaque,
    ) callconv(.c) mlir_capi.LogicalResult.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse
            return c.beaverLogicalResultFailure()));
        return self.invokeVerifier("verify_regions", operation);
    }
};

fn destroyState(comptime State: type) fn (beam.env, ?*anyopaque) callconv(.c) void {
    return struct {
        fn destroy(_: beam.env, object: ?*anyopaque) callconv(.c) void {
            // This runs only after the model-owned reference and every BEAM
            // term reference are gone, so no native callback can still reach
            // the dispatcher.
            const self: *State = @ptrCast(@alignCast(object orelse return));
            self.dispatcher.deinit();
        }
    }.destroy;
}

fn allocateState(
    comptime State: type,
    comptime Dispatcher: type,
    resource_type: beam.resource_type,
    dispatcher: *Dispatcher,
) !*State {
    const memory = e.enif_alloc_resource(resource_type, @sizeOf(State)) orelse
        return error.FailedToAllocateExternalInterfaceState;
    const state: *State = @ptrCast(@alignCast(memory));
    state.* = .{ .dispatcher = dispatcher };
    return state;
}

fn timeout(environment: beam.env, value: beam.term) !u64 {
    return beam.get_u64(environment, value);
}

fn attachSpeculatabilityFallback(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
    const dispatcher = try SpeculatabilityDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try timeout(environment, args[3]),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    dispatcher.setCallback("get_speculatability", args[2]);
    const state = try allocateState(SpeculatabilityState, SpeculatabilityDispatcher, speculatability_state_type, dispatcher);
    errdefer e.enif_release_resource(state);
    dispatcher_owned = false;
    c.mlirConditionallySpeculatableOpInterfaceAttachFallbackModel(
        context,
        try string_ref.get_binary_as_string_ref(environment, args[1]),
        .{
            .construct = SpeculatabilityState.construct,
            .destruct = SpeculatabilityState.destruct,
            .getSpeculatability = SpeculatabilityState.getSpeculatability,
            .userData = state,
        },
    );
    return attachmentTerm(
        SpeculatabilityState,
        environment,
        dispatcher.copyId(environment),
        state,
    );
}

fn attachMemoryEffectsFallback(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
    const dispatcher = try MemoryEffectsDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try timeout(environment, args[3]),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    dispatcher.setCallback("get_effects", args[2]);
    const state = try allocateState(MemoryEffectsState, MemoryEffectsDispatcher, memory_effects_state_type, dispatcher);
    errdefer e.enif_release_resource(state);
    dispatcher_owned = false;
    c.mlirMemoryEffectsOpInterfaceAttachFallbackModel(
        context,
        try string_ref.get_binary_as_string_ref(environment, args[1]),
        .{
            .construct = MemoryEffectsState.construct,
            .destruct = MemoryEffectsState.destruct,
            .getEffects = MemoryEffectsState.getEffects,
            .userData = state,
        },
    );
    return attachmentTerm(
        MemoryEffectsState,
        environment,
        dispatcher.copyId(environment),
        state,
    );
}

fn attachTransformFallback(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
    const dispatcher = try TransformDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try timeout(environment, args[4]),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    dispatcher.setCallback("apply", args[2]);
    if (!beam.is_nil2(environment, args[3]))
        dispatcher.setCallback("allows_repeated_handle_operands", args[3]);
    const state = try allocateState(TransformState, TransformDispatcher, transform_state_type, dispatcher);
    errdefer e.enif_release_resource(state);
    dispatcher_owned = false;
    c.mlirTransformOpInterfaceAttachFallbackModel(
        context,
        try string_ref.get_binary_as_string_ref(environment, args[1]),
        .{
            .construct = TransformState.construct,
            .destruct = TransformState.destruct,
            .apply = TransformState.apply,
            .allowsRepeatedHandleOperands = TransformState.allowsRepeatedHandleOperands,
            .userData = state,
        },
    );
    return attachmentTerm(
        TransformState,
        environment,
        dispatcher.copyId(environment),
        state,
    );
}

fn attachPatternDescriptorFallback(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
    const dispatcher = try PatternDescriptorDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try timeout(environment, args[4]),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    dispatcher.setCallback("populate_patterns", args[2]);
    if (!beam.is_nil2(environment, args[3]))
        dispatcher.setCallback("populate_patterns_with_state", args[3]);
    const state = try allocateState(PatternDescriptorState, PatternDescriptorDispatcher, pattern_descriptor_state_type, dispatcher);
    errdefer e.enif_release_resource(state);
    dispatcher_owned = false;
    c.mlirPatternDescriptorOpInterfaceAttachFallbackModel(
        context,
        try string_ref.get_binary_as_string_ref(environment, args[1]),
        .{
            .construct = PatternDescriptorState.construct,
            .destruct = PatternDescriptorState.destruct,
            .populatePatterns = PatternDescriptorState.populatePatterns,
            .populatePatternsWithState = if (dispatcher.hasCallback("populate_patterns_with_state"))
                PatternDescriptorState.populatePatternsWithState
            else
                null,
            .userData = state,
        },
    );
    return attachmentTerm(
        PatternDescriptorState,
        environment,
        dispatcher.copyId(environment),
        state,
    );
}

fn attachDynamicTrait(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
    const operation_name = try string_ref.get_binary_as_string_ref(environment, args[1]);
    const type_id = try mlir_capi.TypeID.resource.fetch(environment, args[2]);
    const dispatcher = try DynamicTraitDispatcher.initWithOptions(try beam.self(environment), .{
        .timeout_ms = try timeout(environment, args[5]),
    });
    var dispatcher_owned = true;
    errdefer if (dispatcher_owned) dispatcher.deinit();
    if (!beam.is_nil2(environment, args[3])) dispatcher.setCallback("verify", args[3]);
    if (!beam.is_nil2(environment, args[4])) dispatcher.setCallback("verify_regions", args[4]);
    const state = try allocateState(
        DynamicTraitState,
        DynamicTraitDispatcher,
        dynamic_trait_state_type,
        dispatcher,
    );
    errdefer if (dispatcher_owned) e.enif_release_resource(state);
    dispatcher_owned = false;

    // Add the BEAM-term reference before handing the allocation reference to
    // MLIR. A failed insertion destroys the trait synchronously; the term then
    // keeps the state alive until this NIF returns and its environment clears.
    const result = attachmentTerm(
        DynamicTraitState,
        environment,
        dispatcher.copyId(environment),
        state,
    );
    const trait = c.mlirDynamicOpTraitCreate(
        type_id,
        .{
            .construct = DynamicTraitState.construct,
            .destruct = DynamicTraitState.destruct,
            .verifyTrait = DynamicTraitState.verify,
            .verifyRegionTrait = DynamicTraitState.verifyRegions,
        },
        state,
    );

    if (!c.mlirDynamicOpTraitAttach(trait, operation_name, context))
        return error.DynamicTraitAlreadyAttached;
    return result;
}

fn AsyncDiagnosticsNIF(comptime name: []const u8) type {
    const bang = kinda.BangFunc(@import("prelude.zig").allKinds, c, name);

    return struct {
        const Worker = struct {
            recipient: beam.pid,
            environment: beam.env,
            context: mlir_capi.Context.T,
            args: [bang.arity]beam.term,

            fn deinit(self: *@This()) void {
                e.enif_free_env(self.environment);
                std.heap.smp_allocator.destroy(self);
            }

            fn run(user_data: ?*anyopaque) callconv(.c) void {
                const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
                const result = diagnostic.call_with_diagnostics(
                    self.environment,
                    self.context,
                    bang.nif,
                    .{ self.environment, bang.arity, &self.args },
                ) catch {
                    self.deinit();
                    return;
                };

                _ = beam.send_advanced(null, self.recipient, self.environment, result);
                self.deinit();
            }
        };

        fn nif(environment: beam.env, n: c_int, args: [*c]const beam.term) !beam.term {
            if (n != bang.arity + 1) return error.BadArity;
            const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
            const worker = try std.heap.smp_allocator.create(Worker);
            errdefer std.heap.smp_allocator.destroy(worker);
            const owned_environment = e.enif_alloc_env() orelse
                return error.FailedToAllocateDiagnosticEnvironment;
            errdefer e.enif_free_env(owned_environment);

            worker.* = .{
                .recipient = try beam.self(environment),
                .environment = owned_environment,
                .context = context,
                .args = undefined,
            };
            for (0..bang.arity) |index| {
                worker.args[index] = e.enif_make_copy(owned_environment, args[index + 1]);
            }

            if (!c.beaverContextAddWork(context, Worker.run, worker)) {
                return error.ContextMultithreadingDisabled;
            }
            return beam.make_atom(environment, "async");
        }
    };
}

fn releaseStateTerm(
    comptime State: type,
    environment: beam.env,
    resource_type: beam.resource_type,
    term: beam.term,
) bool {
    var object: ?*anyopaque = null;
    if (e.enif_get_resource(environment, term, resource_type, @ptrCast(&object)) == 0)
        return false;
    const state: *State = @ptrCast(@alignCast(object orelse return false));
    releaseModel(State, state);
    return true;
}

fn releaseExternalInterface(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const released =
        releaseStateTerm(SpeculatabilityState, environment, speculatability_state_type, args[0]) or
        releaseStateTerm(MemoryEffectsState, environment, memory_effects_state_type, args[0]) or
        releaseStateTerm(TransformState, environment, transform_state_type, args[0]) or
        releaseStateTerm(PatternDescriptorState, environment, pattern_descriptor_state_type, args[0]) or
        releaseStateTerm(DynamicTraitState, environment, dynamic_trait_state_type, args[0]);
    if (!released) return error.InvalidExternalInterfaceState;
    return beam.make_ok(environment);
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
    if (!c.beaverContextAddWork(context, SpeculatabilityWorker.run, worker))
        return error.ContextMultithreadingDisabled;
    return beam.make_ok(environment);
}

fn collectTransformState(
    comptime Kind: type,
    comptime Iterator: anytype,
    environment: beam.env,
    args: [*c]const beam.term,
) !beam.term {
    const state = try mlir_capi.TransformState.resource.fetch(environment, args[0]);
    const value = try mlir_capi.Value.resource.fetch(environment, args[1]);
    var values = std.array_list.Managed(Kind.T).init(std.heap.smp_allocator);
    defer values.deinit();

    const Collector = struct {
        fn append(item: Kind.T, user_data: ?*anyopaque) callconv(.c) void {
            const list: *std.array_list.Managed(Kind.T) = @ptrCast(@alignCast(user_data orelse return));
            list.append(item) catch @panic("failed to collect transform state mapping");
        }
    };

    Iterator(state, value, Collector.append, &values);
    return kinda.callback_adapter.handleRange(Kind, environment, values.items);
}

fn transformStatePayloadOps(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return collectTransformState(
        mlir_capi.Operation,
        c.mlirTransformStateForEachPayloadOp,
        environment,
        args,
    );
}

fn transformStatePayloadValues(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return collectTransformState(
        mlir_capi.Value,
        c.mlirTransformStateForEachPayloadValue,
        environment,
        args,
    );
}

fn transformStateParams(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return collectTransformState(
        mlir_capi.Attribute,
        c.mlirTransformStateForEachParam,
        environment,
        args,
    );
}

fn openResourceType(
    environment: beam.env,
    comptime name: [*c]const u8,
    destructor: e.ErlNifResourceDtor,
) beam.resource_type {
    const resource_type = e.enif_open_resource_type(
        environment,
        null,
        name,
        destructor,
        e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER,
        null,
    );
    if (resource_type == null) @panic("failed to open external interface state resource");
    return resource_type;
}

pub fn open(environment: beam.env) void {
    speculatability_state_type = openResourceType(
        environment,
        "Beaver.MLIR.ConditionallySpeculatable.FallbackState",
        destroyState(SpeculatabilityState),
    );
    memory_effects_state_type = openResourceType(
        environment,
        "Beaver.MLIR.MemoryEffects.FallbackState",
        destroyState(MemoryEffectsState),
    );
    transform_state_type = openResourceType(
        environment,
        "Beaver.MLIR.TransformOpInterface.FallbackState",
        destroyState(TransformState),
    );
    pattern_descriptor_state_type = openResourceType(
        environment,
        "Beaver.MLIR.PatternDescriptorOpInterface.FallbackState",
        destroyState(PatternDescriptorState),
    );
    dynamic_trait_state_type = openResourceType(
        environment,
        "Beaver.MLIR.Trait.DynamicState",
        destroyState(DynamicTraitState),
    );
}

pub const nifs = .{
    prelude.beaverRawNIF(@This(), "conditionally_speculatable_attach_fallback_model", 4),
    prelude.beaverRawNIF(@This(), "conditionally_speculatable_query_async", 1),
    prelude.beaverRawNIF(@This(), "memory_effects_attach_fallback_model", 4),
    prelude.beaverRawNIF(@This(), "transform_op_interface_attach_fallback_model", 5),
    prelude.beaverRawNIF(@This(), "pattern_descriptor_op_interface_attach_fallback_model", 5),
    prelude.beaverRawNIF(@This(), "dynamic_trait_attach", 6),
    prelude.beaverRawNIF(@This(), "module_create_parse_async", 3),
    prelude.beaverRawNIF(@This(), "operation_verify_async", 2),
    prelude.beaverRawNIF(@This(), "transform_state_payload_ops", 2),
    prelude.beaverRawNIF(@This(), "transform_state_payload_values", 2),
    prelude.beaverRawNIF(@This(), "transform_state_params", 2),
    prelude.beaverRawNIF(@This(), "external_interface_release", 1),
    kinda.callback_runtime.ReplyToken.codeNif("beaver_raw_callback_reply_code"),
};

pub const conditionally_speculatable_attach_fallback_model = attachSpeculatabilityFallback;
pub const conditionally_speculatable_query_async = querySpeculatabilityAsync;
pub const memory_effects_attach_fallback_model = attachMemoryEffectsFallback;
pub const transform_op_interface_attach_fallback_model = attachTransformFallback;
pub const pattern_descriptor_op_interface_attach_fallback_model = attachPatternDescriptorFallback;
pub const dynamic_trait_attach = attachDynamicTrait;
pub const module_create_parse_async = AsyncDiagnosticsNIF("mlirModuleCreateParse").nif;
pub const operation_verify_async = AsyncDiagnosticsNIF("mlirOperationVerify").nif;
pub const transform_state_payload_ops = transformStatePayloadOps;
pub const transform_state_payload_values = transformStatePayloadValues;
pub const transform_state_params = transformStateParams;
pub const external_interface_release = releaseExternalInterface;
