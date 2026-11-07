const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const mlir_capi = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");
const callback_args = @import("callback_args.zig");

const SpeculatabilityDispatcher = kinda.callback_runtime.Dispatcher(.{"get_speculatability"});
const MemoryEffectsDispatcher = kinda.callback_runtime.Dispatcher(.{"get_effects"});
const TransformDispatcher = kinda.callback_runtime.Dispatcher(.{ "apply", "allows_repeated_handle_operands" });
const PatternDescriptorDispatcher = kinda.callback_runtime.Dispatcher(.{ "populate_patterns", "populate_patterns_with_state" });
const DynamicTraitDispatcher = kinda.callback_runtime.Dispatcher(.{ "verify", "verify_regions" });

fn notifyReleased(recipient: beam.pid) void {
    const environment = e.enif_alloc_env() orelse return;
    defer e.enif_free_env(environment);
    _ = beam.send_advanced(null, recipient, environment, beam.make_atom(environment, "external_interface_released"));
}

fn releaseModel(comptime State: type, self: *State) void {
    if (self.model_released.cmpxchgStrong(false, true, .acq_rel, .acquire) == null)
        e.enif_release_resource(self);
}

fn attachmentTerm(comptime State: type, environment: beam.env, dispatcher_id: beam.term, state: *State) beam.term {
    // Keep the reference returned by enif_alloc_resource for the MLIR model.
    // enif_make_resource adds an independent BEAM-term reference. If the
    // attachment process dies and drops that term, the model reference keeps
    // both this state and its dispatcher alive until the model destructor calls
    // releaseModel. Do not release the allocation reference here.
    var terms = [_]beam.term{ dispatcher_id, e.enif_make_resource(environment, state) };
    return beam.make_tuple(environment, &terms);
}

const SpeculatabilityState = struct {
    const resource_name = "Beaver.MLIR.ConditionallySpeculatable.FallbackState";
    dispatcher: *SpeculatabilityDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn getSpeculatability(operation: mlir_capi.Operation.T, user_data: ?*anyopaque) callconv(.c) c.MlirSpeculatability {
        const failure = c.MlirSpeculatabilityNotSpeculatable;
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return failure));
        const response = callback_args.invoke(self.dispatcher, "get_speculatability", .{mlir_capi.Operation}, .{operation}) catch return failure;
        return kinda.callback_adapter.enumResult(c.MlirSpeculatability, response, failure);
    }
};

const MemoryEffectsState = struct {
    const resource_name = "Beaver.MLIR.MemoryEffects.FallbackState";
    dispatcher: *MemoryEffectsDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn appendConservativeEffect(effects: mlir_capi.MemoryEffectInstancesList.T) void {
        const instance = c.mlirMemoryEffectInstanceCreate(
            c.mlirMemoryEffectsWriteGet(),
            c.mlirAttributeGetNull(),
            0,
            false,
            c.mlirSideEffectsDefaultResourceGet(),
        );
        defer c.mlirMemoryEffectInstanceDestroy(instance);
        c.beaverMemoryEffectInstancesListAppend(effects, instance);
    }

    fn getEffects(operation: mlir_capi.Operation.T, callback: c.MlirMemoryEffectInstancesCallback, callback_data: ?*anyopaque, user_data: ?*anyopaque) callconv(.c) void {
        var sink = c.MlirBeaverMemoryEffectSink{ .callback = callback, .userData = callback_data };
        const effects = mlir_capi.MemoryEffectInstancesList.T{ .ptr = &sink };
        const self: *@This() = @ptrCast(@alignCast(user_data orelse {
            appendConservativeEffect(effects);
            return;
        }));
        const response = callback_args.invoke(self.dispatcher, "get_effects", .{ mlir_capi.Operation, mlir_capi.MemoryEffectInstancesList }, .{ operation, effects }) catch {
            appendConservativeEffect(effects);
            return;
        };
        if (!response.success) appendConservativeEffect(effects);
    }
};

const TransformState = struct {
    const resource_name = "Beaver.MLIR.TransformOpInterface.FallbackState";
    dispatcher: *TransformDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn apply(operation: mlir_capi.Operation.T, rewriter: mlir_capi.TransformRewriter.T, results: mlir_capi.TransformResults.T, state: mlir_capi.TransformState.T, user_data: ?*anyopaque) callconv(.c) c.MlirDiagnosedSilenceableFailure {
        const failure = c.MlirDiagnosedSilenceableFailureDefiniteFailure;
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return failure));
        const response = callback_args.invoke(self.dispatcher, "apply", .{ mlir_capi.Operation, mlir_capi.TransformRewriter, mlir_capi.TransformResults, mlir_capi.TransformState }, .{ operation, rewriter, results, state }) catch return failure;
        return kinda.callback_adapter.enumResult(c.MlirDiagnosedSilenceableFailure, response, failure);
    }

    fn allowsRepeatedHandleOperands(operation: mlir_capi.Operation.T, user_data: ?*anyopaque) callconv(.c) bool {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return false));
        if (!self.dispatcher.hasCallback("allows_repeated_handle_operands")) return false;
        const response = callback_args.invoke(self.dispatcher, "allows_repeated_handle_operands", .{mlir_capi.Operation}, .{operation}) catch return false;
        return kinda.callback_adapter.scalarResult(u1, response, 0) != 0;
    }
};

const PatternDescriptorState = struct {
    const resource_name = "Beaver.MLIR.PatternDescriptorOpInterface.FallbackState";
    dispatcher: *PatternDescriptorDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn populatePatterns(operation: mlir_capi.Operation.T, patterns: mlir_capi.RewritePatternSet.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        _ = callback_args.invoke(self.dispatcher, "populate_patterns", .{ mlir_capi.Operation, mlir_capi.RewritePatternSet }, .{ operation, patterns }) catch {};
    }

    fn populatePatternsWithState(operation: mlir_capi.Operation.T, patterns: mlir_capi.RewritePatternSet.T, state: mlir_capi.TransformState.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        _ = callback_args.invoke(self.dispatcher, "populate_patterns_with_state", .{ mlir_capi.Operation, mlir_capi.RewritePatternSet, mlir_capi.TransformState }, .{ operation, patterns, state }) catch {};
    }
};

const DynamicTraitState = struct {
    const resource_name = "Beaver.MLIR.Trait.DynamicState";
    dispatcher: *DynamicTraitDispatcher,
    model_released: std.atomic.Value(bool) = .init(false),

    fn invokeVerifier(self: *@This(), comptime callback: []const u8, operation: mlir_capi.Operation.T) mlir_capi.LogicalResult.T {
        if (!self.dispatcher.hasCallback(callback)) return c.beaverLogicalResultSuccess();
        const response = callback_args.invoke(self.dispatcher, callback, .{mlir_capi.Operation}, .{operation}) catch return callbackFailure(operation);
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

    fn verify(operation: mlir_capi.Operation.T, user_data: ?*anyopaque) callconv(.c) mlir_capi.LogicalResult.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return c.beaverLogicalResultFailure()));
        return self.invokeVerifier("verify", operation);
    }

    fn verifyRegions(operation: mlir_capi.Operation.T, user_data: ?*anyopaque) callconv(.c) mlir_capi.LogicalResult.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return c.beaverLogicalResultFailure()));
        return self.invokeVerifier("verify_regions", operation);
    }
};

fn constructModel(_: ?*anyopaque) callconv(.c) void {}

fn destructModel(comptime State: type) fn (?*anyopaque) callconv(.c) void {
    return struct {
        fn destruct(object: ?*anyopaque) callconv(.c) void {
            const self: *State = @ptrCast(@alignCast(object orelse return));
            notifyReleased(self.dispatcher.handler);
            releaseModel(State, self);
        }
    }.destruct;
}

fn stateResource(comptime State: type) type {
    return kinda.RawResourceType(State, State.resource_name, destroyState(State));
}

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

fn allocateState(comptime State: type, comptime callbacks: anytype, comptime required: usize, environment: beam.env, args: []const beam.term) !*State {
    const Dispatcher = @typeInfo(@FieldType(State, "dispatcher")).pointer.child;
    const dispatcher = try Dispatcher.initWithOptions(try beam.self(environment), callback_args.fixedOptions(Dispatcher, try beam.get_u64(environment, args[callbacks.len])));
    errdefer dispatcher.deinit();
    inline for (callbacks, 0..) |name, index| {
        if (index < required or !beam.is_nil2(environment, args[index])) dispatcher.setCallback(name, args[index]);
    }
    const state = stateResource(State).alloc() catch return error.FailedToAllocateExternalInterfaceState;
    state.* = .{ .dispatcher = dispatcher };
    return state;
}

// Every fallback transfers the allocation reference to MLIR and returns a
// separate BEAM reference. Only the state-aware descriptor hook is nullable.
fn Fallback(comptime State: type, comptime callbacks: anytype, comptime Attach: anytype, comptime methods: anytype) type {
    return struct {
        fn attach(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
            const name = try string_ref.get_binary_as_string_ref(environment, args[1]);
            const state = try allocateState(State, callbacks, 1, environment, args[2 .. 3 + callbacks.len]);
            var model: @typeInfo(@TypeOf(Attach)).@"fn".params[2].type.? = undefined;
            model.construct = constructModel;
            model.destruct = destructModel(State);
            model.userData = state;
            inline for (std.meta.fields(@TypeOf(methods))) |field| @field(model, field.name) = @field(methods, field.name);
            if (State == PatternDescriptorState and !state.dispatcher.hasCallback("populate_patterns_with_state"))
                model.populatePatternsWithState = null;
            c.beaverExecutionMarkCallbacks(context);
            Attach(context, name, model);
            return attachmentTerm(State, environment, state.dispatcher.copyId(environment), state);
        }
    };
}

fn attachDynamicTrait(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(environment, args[0]);
    c.beaverExecutionMarkCallbacks(context);
    const operation_name = try string_ref.get_binary_as_string_ref(environment, args[1]);
    const type_id = try mlir_capi.TypeID.resource.fetch(environment, args[2]);
    const state = try allocateState(DynamicTraitState, .{ "verify", "verify_regions" }, 0, environment, args[3..6]);

    // Add the BEAM-term reference before handing the allocation reference to
    // MLIR. A failed insertion destroys the trait synchronously; the term then
    // keeps the state alive until this NIF returns and its environment clears.
    const result = attachmentTerm(DynamicTraitState, environment, state.dispatcher.copyId(environment), state);
    const trait = c.mlirDynamicOpTraitCreate(
        type_id,
        .{
            .construct = constructModel,
            .destruct = destructModel(DynamicTraitState),
            .verifyTrait = DynamicTraitState.verify,
            .verifyRegionTrait = DynamicTraitState.verifyRegions,
        },
        state,
    );

    if (!c.mlirDynamicOpTraitAttach(trait, operation_name, context))
        return error.DynamicTraitAlreadyAttached;
    return result;
}

const States = .{ SpeculatabilityState, MemoryEffectsState, TransformState, PatternDescriptorState, DynamicTraitState };

fn releaseExternalInterface(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    inline for (States) |State| {
        if (stateResource(State).fetch(environment, args[0])) |state| {
            releaseModel(State, state);
            return beam.make_ok(environment);
        } else |_| {}
    }
    return error.InvalidExternalInterfaceState;
}

const EffectsCollector = struct {
    environment: beam.env,
    terms: std.array_list.Managed(beam.term),
    failed: bool = false,

    fn append(count: isize, instances: [*c]c.MlirMemoryEffectInstance, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        if (count < 0) {
            self.failed = true;
            return;
        }
        for (0..@intCast(count)) |index| self.instance(instances[index]) catch {
            self.failed = true;
            return;
        };
    }

    fn instance(self: *@This(), effect: c.MlirMemoryEffectInstance) !void {
        var target_kind = beam.make_atom(self.environment, "none");
        var target = beam.make_nil(self.environment);
        inline for (.{
            .{ "operand", mlir_capi.OpOperand, c.beaverMemoryEffectInstanceGetOpOperand(effect) },
            .{ "value", mlir_capi.Value, c.mlirMemoryEffectInstanceGetValue(effect) },
            .{ "symbol", mlir_capi.Attribute, c.mlirMemoryEffectInstanceGetSymbolRef(effect) },
        }) |entry| {
            if (entry[2].ptr != null) {
                target_kind = beam.make_atom(self.environment, entry[0]);
                target = try kinda.callback_adapter.handle(entry[1], self.environment, entry[2]);
                break;
            }
        }
        var fields = [_]beam.term{
            beam.make_c_int(self.environment, @intCast(c.beaverMemoryEffectInstanceGetKind(effect))),
            target_kind,
            target,
            try kinda.callback_adapter.handle(mlir_capi.Attribute, self.environment, c.mlirMemoryEffectInstanceGetParameters(effect)),
            beam.make_c_int(self.environment, c.mlirMemoryEffectInstanceGetStage(effect)),
            beam.make_bool(self.environment, c.mlirMemoryEffectInstanceGetEffectOnFullRegion(effect)),
            try kinda.callback_adapter.handle(mlir_capi.SideEffectResource, self.environment, c.mlirMemoryEffectInstanceGetResource(effect)),
        };
        try self.terms.append(beam.make_tuple(self.environment, &fields));
    }
};

pub fn memoryEffects(environment: beam.env, operation: mlir_capi.Operation.T) !beam.term {
    var collector = EffectsCollector{ .environment = environment, .terms = .init(std.heap.smp_allocator) };
    defer collector.terms.deinit();
    if (!c.mlirOperationImplementsInterface(operation, c.mlirMemoryEffectsOpInterfaceTypeID())) return beam.make_atom(environment, "error");
    c.mlirMemoryEffectsOpInterfaceGetEffects(operation, EffectsCollector.append, &collector);
    return if (!collector.failed) beam.make_term_list(environment, collector.terms.items) else beam.make_atom(environment, "error");
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
    return collectTransformState(mlir_capi.Operation, c.mlirTransformStateForEachPayloadOp, environment, args);
}

fn transformStatePayloadValues(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return collectTransformState(mlir_capi.Value, c.mlirTransformStateForEachPayloadValue, environment, args);
}

fn transformStateParams(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return collectTransformState(mlir_capi.Attribute, c.mlirTransformStateForEachParam, environment, args);
}

pub fn open(environment: beam.env) void {
    inline for (States) |State| stateResource(State).open(environment);
}

pub const nifs = .{
    prelude.beaverRawNIF(@This(), "conditionally_speculatable_attach_fallback_model", 4),
    prelude.beaverRawNIF(@This(), "memory_effects_attach_fallback_model", 4),
    prelude.beaverRawNIF(@This(), "transform_op_interface_attach_fallback_model", 5),
    prelude.beaverRawNIF(@This(), "pattern_descriptor_op_interface_attach_fallback_model", 5),
    prelude.beaverRawNIF(@This(), "dynamic_trait_attach", 6),
    prelude.beaverRawNIF(@This(), "transform_state_payload_ops", 2),
    prelude.beaverRawNIF(@This(), "transform_state_payload_values", 2),
    prelude.beaverRawNIF(@This(), "transform_state_params", 2),
    prelude.beaverRawNIF(@This(), "external_interface_release", 1),
    kinda.callback_runtime.ReplyToken.codeNif("beaver_raw_callback_reply_code"),
};

pub const conditionally_speculatable_attach_fallback_model = Fallback(
    SpeculatabilityState,
    .{"get_speculatability"},
    c.mlirConditionallySpeculatableOpInterfaceAttachFallbackModel,
    .{ .getSpeculatability = SpeculatabilityState.getSpeculatability },
).attach;
pub const memory_effects_attach_fallback_model = Fallback(
    MemoryEffectsState,
    .{"get_effects"},
    c.mlirMemoryEffectsOpInterfaceAttachFallbackModel,
    .{ .getEffects = MemoryEffectsState.getEffects },
).attach;
pub const transform_op_interface_attach_fallback_model = Fallback(TransformState, .{ "apply", "allows_repeated_handle_operands" }, c.mlirTransformOpInterfaceAttachFallbackModel, .{
    .apply = TransformState.apply,
    .allowsRepeatedHandleOperands = TransformState.allowsRepeatedHandleOperands,
}).attach;
pub const pattern_descriptor_op_interface_attach_fallback_model = Fallback(PatternDescriptorState, .{ "populate_patterns", "populate_patterns_with_state" }, c.mlirPatternDescriptorOpInterfaceAttachFallbackModel, .{
    .populatePatterns = PatternDescriptorState.populatePatterns,
    .populatePatternsWithState = PatternDescriptorState.populatePatternsWithState,
}).attach;
pub const dynamic_trait_attach = attachDynamicTrait;
pub const transform_state_payload_ops = transformStatePayloadOps;
pub const transform_state_payload_values = transformStatePayloadValues;
pub const transform_state_params = transformStateParams;
pub const external_interface_release = releaseExternalInterface;
