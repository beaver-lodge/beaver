const std = @import("std");
const builtin = @import("builtin");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const mlir_capi = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");
const diagnostic = @import("diagnostic.zig");
const callback_args = @import("callback_args.zig");
const Values = callback_args.Range(mlir_capi.Value);
const Types = callback_args.Range(mlir_capi.Type);
const OptionalType = callback_args.Optional(mlir_capi.Type);
const Timestamp = struct {
    pub fn encode(environment: beam.env, _: void) !beam.term {
        return beam.make_i64(environment, systemNs());
    }
};
const LegalityDispatcher = kinda.callback_runtime.Dispatcher(.{"conversion_legality"});
const TypeDispatcher = kinda.callback_runtime.Dispatcher(.{
    "convert_type",
    "convert_types",
    "source_materialization",
    "target_materialization",
    "target_materialization_1_to_n",
});
const PatternDispatcher = kinda.callback_runtime.Dispatcher(.{ "conversion_pattern", "conversion_pattern_1_to_n" });
const OperationInventory = struct {
    operations: u64 = 0,
    modules: u64 = 0,
    functions: u64 = 0,
    fn walk(operation: mlir_capi.Operation.T, user_data: ?*anyopaque) callconv(.c) c.MlirWalkResult {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return c.MlirWalkResultInterrupt));
        self.operations += 1;
        const name = c.mlirIdentifierStr(c.mlirOperationGetName(operation));
        const bytes = name.data[0..name.length];
        if (std.mem.eql(u8, bytes, "builtin.module")) self.modules += 1;
        if (std.mem.eql(u8, bytes, "func.func")) self.functions += 1;
        return c.MlirWalkResultAdvance;
    }
    pub fn operation_inventory(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const operation = try mlir_capi.Operation.resource.fetch(environment, args[0]);
        var inventory: @This() = .{};
        c.mlirOperationWalk(operation, walk, &inventory, c.MlirWalkPreOrder);
        var fields = [_]beam.term{
            beam.make_u64(environment, inventory.operations),
            beam.make_u64(environment, inventory.modules),
            beam.make_u64(environment, inventory.functions),
        };
        return beam.make_tuple(environment, &fields);
    }
};
const ProfileMetrics = struct {
    pub fn process_cpu_time(environment: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
        const timestamp_ns = std.Io.Clock.cpu_process.now(std.Options.debug_io).toNanoseconds();
        const value = std.math.cast(u64, timestamp_ns) orelse 0;
        return beam.make_u64(environment, value);
    }
    pub fn peak_rss(environment: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
        const bytes: u64 = switch (builtin.os.tag) {
            .windows, .wasi, .freestanding => 0,
            else => value: {
                const usage = std.posix.getrusage(std.posix.rusage.SELF);
                const max_rss: u64 = if (usage.maxrss <= 0) 0 else @intCast(usage.maxrss);
                break :value switch (builtin.os.tag) {
                    .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => max_rss,
                    else => std.math.mul(u64, max_rss, 1024) catch std.math.maxInt(u64),
                };
            },
        };
        return beam.make_u64(environment, bytes);
    }
};
fn systemNs() i64 {
    // The C++ clock is valid on scheduler and arbitrary worker threads.
    return @intCast(c.beaverExecutionClock());
}
fn HandleProjection(comptime Kind: type) type {
    return struct {
        values: []Kind.T,
        fn init(environment: beam.env, list: beam.term) !*@This() {
            const self = try std.heap.smp_allocator.create(@This());
            errdefer std.heap.smp_allocator.destroy(self);
            self.* = .{ .values = try callback_args.Range(Kind).decode(environment, list, std.heap.smp_allocator) };
            return self;
        }
        fn deinit(self: *@This()) void {
            std.heap.smp_allocator.free(self.values);
            std.heap.smp_allocator.destroy(self);
        }
    };
}
fn replyHandle(comptime Kind: type, comptime many: bool) type {
    return struct {
        fn nif(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const scalar_value = Kind == mlir_capi.Value and !many;
            const success = try beam.get_bool(environment, args[1]);
            const code = if (scalar_value) 0 else try beam.get_i64(environment, args[2]);
            const term = args[if (scalar_value) 2 else 3];
            var projection: usize = 0;
            var owned: ?*HandleProjection(Kind) = null;
            errdefer if (owned) |value| value.deinit();
            if (success and code == c.MlirTypeConverterConversionStatusSuccess) {
                if (many) {
                    owned = try HandleProjection(Kind).init(environment, term);
                    projection = @intFromPtr(owned.?);
                } else if (!(scalar_value and beam.is_nil2(environment, term))) {
                    projection = @intFromPtr((try Kind.resource.fetch(environment, term)).ptr);
                }
            }
            const accepted = try kinda.callback_runtime.ReplyToken.complete(environment, args[0], .replied, success, code, projection);
            if (!accepted) if (owned) |value| value.deinit();
            return beam.make_atom(environment, if (accepted) "ok" else "stale");
        }
    };
}
const LegalityState = struct {
    dispatcher: *LegalityDispatcher,
    fn deinit(self: *@This()) void {
        self.dispatcher.deinit();
        std.heap.smp_allocator.destroy(self);
    }
    fn callback(operation: mlir_capi.Operation.T, user_data: ?*anyopaque) callconv(.c) c.MlirConversionTargetLegality {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return c.MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL));
        const response = callback_args.invoke(
            self.dispatcher,
            "conversion_legality",
            .{ Timestamp, mlir_capi.Operation },
            .{ {}, operation },
        ) catch return c.MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL;
        return kinda.callback_adapter.scalarResult(
            c.MlirConversionTargetLegality,
            response,
            c.MLIR_CONVERSION_TARGET_LEGALITY_ILLEGAL,
        );
    }
};
// Targets and converters share the same serialized lifetime. Pattern-owned
// references keep converters alive; explicit close still rejects active users.
fn Registration(comptime Kind: type, comptime Callback: type, comptime name: []const u8, comptime tag: []const u8) type {
    return struct {
        const Self = @This();
        // Do not take over older resources with the pre-unification layout.
        const Resource = kinda.RawResourceType(Self, "Beaver.MLIR." ++ name ++ ".Registration.V2", destruct);
        handle: Kind.T,
        context: mlir_capi.Context.T = .{ .ptr = null },
        callbacks: std.array_list.Managed(*Callback) = .init(std.heap.smp_allocator),
        mappings: std.array_list.Managed(*TypeMappingState) = .init(std.heap.smp_allocator),
        mutex: std.Io.Mutex = .init,
        closed: std.atomic.Value(bool) = .init(false),
        pattern_users: std.atomic.Value(usize) = .init(0),
        fn closeLocked(self: *Self) bool {
            if (self.closed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return false;
            if (Kind == mlir_capi.ConversionTarget) c.mlirConversionTargetDestroy(self.handle) else c.mlirTypeConverterDestroy(self.handle);
            inline for (.{ "callbacks", "mappings" }) |field| {
                for (@field(self, field).items) |item| item.deinit();
                @field(self, field).deinit();
            }
            return true;
        }
        fn close(self: *Self) bool {
            self.mutex.lockUncancelable(std.Options.debug_io);
            defer self.mutex.unlock(std.Options.debug_io);
            return self.closeLocked();
        }
        fn destruct(_: beam.env, object: ?*anyopaque) callconv(.c) void {
            const self: *Self = @ptrCast(@alignCast(object orelse return));
            _ = self.close();
        }
        fn create(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const context = if (Kind == mlir_capi.ConversionTarget) try mlir_capi.Context.resource.fetch(environment, args[0]) else mlir_capi.Context.T{ .ptr = null };
            const self = Resource.alloc() catch return @field(anyerror, "FailedToAllocate" ++ name ++ "Registration");
            self.* = .{
                .handle = if (Kind == mlir_capi.ConversionTarget) c.mlirConversionTargetCreate(context) else c.mlirTypeConverterCreate(),
                .context = context,
            };
            defer Resource.release(self);
            var terms = [_]beam.term{ beam.make_atom(environment, tag), try Kind.resource.make_kind(environment, self.handle), e.enif_make_resource(environment, self) };
            return beam.make_tuple(environment, &terms);
        }
        fn destroy(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const self = try Resource.fetch(environment, args[0]);
            self.mutex.lockUncancelable(std.Options.debug_io);
            defer self.mutex.unlock(std.Options.debug_io);
            if (self.pattern_users.load(.acquire) != 0) return error.TypeConverterInUseByPatterns;
            _ = self.closeLocked();
            return beam.make_ok(environment);
        }
    };
}
const TargetRegistration = Registration(mlir_capi.ConversionTarget, LegalityState, "ConversionTarget", "managed_conversion_target");
const TypeConverterRegistration = Registration(mlir_capi.TypeConverter, TypeCallbackState, "TypeConverter", "managed_type_converter");
const TargetRegistrationResource = TargetRegistration.Resource;
const TypeConverterRegistrationResource = TypeConverterRegistration.Resource;
const fetchTargetRegistration = TargetRegistrationResource.fetch;
const fetchTypeConverterRegistration = TypeConverterRegistrationResource.fetch;
fn appendCallback(comptime State: type, environment: beam.env, callbacks: *std.array_list.Managed(*State), comptime name: []const u8, callback: beam.term, timeout: beam.term) !*State {
    const Dispatcher = @typeInfo(@FieldType(State, "dispatcher")).pointer.child;
    const dispatcher = try Dispatcher.initWithOptions(try beam.self(environment), callback_args.fixedOptions(Dispatcher, try beam.get_u64(environment, timeout)));
    errdefer dispatcher.deinit();
    dispatcher.setCallback(name, callback);
    const state = try std.heap.smp_allocator.create(State);
    errdefer std.heap.smp_allocator.destroy(state);
    state.* = .{ .dispatcher = dispatcher };
    try callbacks.append(state);
    return state;
}
const LegalityKind = enum { operation, dialect, recursively_legal, unknown };
fn Legality(comptime kind: LegalityKind) type {
    return struct {
        fn add(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const registration = try fetchTargetRegistration(environment, args[0]);
            registration.mutex.lockUncancelable(std.Options.debug_io);
            defer registration.mutex.unlock(std.Options.debug_io);
            if (registration.closed.load(.acquire)) return error.ConversionTargetClosed;
            const named = kind != .unknown;
            const callback = args[if (named) 2 else 1];
            const name = if (named) try string_ref.get_binary_as_string_ref(environment, args[1]) else mlir_capi.StringRef.T{ .data = null, .length = 0 };
            if (kind == .recursively_legal and beam.is_nil2(environment, callback)) {
                c.mlirConversionTargetMarkOpRecursivelyLegal(registration.handle, name, null, null);
            } else {
                const state = try appendCallback(LegalityState, environment, &registration.callbacks, "conversion_legality", callback, args[if (named) 3 else 2]);
                if (named) {
                    const register = switch (kind) {
                        .operation => c.mlirConversionTargetAddDynamicallyLegalOp,
                        .dialect => c.mlirConversionTargetAddDynamicallyLegalDialect,
                        .recursively_legal => c.mlirConversionTargetMarkOpRecursivelyLegal,
                        .unknown => unreachable,
                    };
                    register(registration.handle, name, LegalityState.callback, state);
                } else c.mlirConversionTargetMarkUnknownOpDynamicallyLegal(registration.handle, LegalityState.callback, state);
            }
            return beam.make_ok(environment);
        }
    };
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
        0 => c.mlirConversionTargetAddLegalOp(registration.handle, name),
        1 => c.mlirConversionTargetAddIllegalOp(registration.handle, name),
        2 => c.mlirConversionTargetAddLegalDialect(registration.handle, name),
        3 => c.mlirConversionTargetAddIllegalDialect(registration.handle, name),
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
    fn Conversion(comptime many: bool) type {
        const Output = if (many) mlir_capi.TypeConverterConversionResults.T else [*c]mlir_capi.Type.T;
        return struct {
            fn run(type_: mlir_capi.Type.T, output: Output, data: ?*anyopaque) callconv(.c) c.MlirTypeConverterConversionStatus {
                const self: *TypeCallbackState = @ptrCast(@alignCast(data orelse return c.MlirTypeConverterConversionStatusFailure));
                const response = callback_args.invoke(self.dispatcher, if (many) "convert_types" else "convert_type", .{ Timestamp, mlir_capi.Type }, .{ {}, type_ }) catch return c.MlirTypeConverterConversionStatusFailure;
                const status = kinda.callback_adapter.scalarResult(c.MlirTypeConverterConversionStatus, response, c.MlirTypeConverterConversionStatusFailure);
                if (status == c.MlirTypeConverterConversionStatusSuccess) {
                    const projection = kinda.callback_adapter.projection(response) orelse return c.MlirTypeConverterConversionStatusFailure;
                    if (many) {
                        const projected: *HandleProjection(mlir_capi.Type) = @ptrFromInt(projection);
                        defer projected.deinit();
                        for (projected.values) |value| c.mlirTypeConverterConversionResultsAppend(output, value);
                    } else output.* = .{ .ptr = @ptrFromInt(projection) };
                }
                return status;
            }
        };
    }
    const conversion = Conversion(false).run;
    const conversion1ToN = Conversion(true).run;
    fn sourceMaterialization(
        rewriter: mlir_capi.RewriterBase.T,
        output_type: mlir_capi.Type.T,
        n_inputs: isize,
        inputs: [*c]mlir_capi.Value.T,
        location: mlir_capi.Location.T,
        user_data: ?*anyopaque,
    ) callconv(.c) mlir_capi.Value.T {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return .{ .ptr = null }));
        const response = callback_args.invoke(
            self.dispatcher,
            "source_materialization",
            .{ Timestamp, mlir_capi.RewriterBase, mlir_capi.Type, Values, mlir_capi.Location },
            .{ {}, rewriter, output_type, Values.slice(n_inputs, inputs), location },
        ) catch return .{ .ptr = null };
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
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return .{ .ptr = null }));
        const response = callback_args.invoke(
            self.dispatcher,
            "target_materialization",
            .{ Timestamp, mlir_capi.RewriterBase, mlir_capi.Type, Values, mlir_capi.Location, mlir_capi.Type },
            .{ {}, rewriter, output_type, Values.slice(n_inputs, inputs), location, original_type },
        ) catch return .{ .ptr = null };
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
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return c.beaverLogicalResultFailure()));
        const response = callback_args.invoke(
            self.dispatcher,
            "target_materialization_1_to_n",
            .{ Timestamp, mlir_capi.RewriterBase, Types, Values, mlir_capi.Location, OptionalType },
            .{ {}, rewriter, Types.slice(n_output_types, output_types), Values.slice(n_inputs, inputs), location, original_type },
        ) catch return c.beaverLogicalResultFailure();
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
extern fn beaverConvertCompositeType(mlir_capi.Type.T, [*]const mlir_capi.Type.T, usize, mlir_capi.Type.T) mlir_capi.Type.T;
extern fn beaverPopulateStructuralTypeConversions(mlir_capi.ConversionTarget.T, mlir_capi.RewritePatternSet.T, mlir_capi.TypeConverter.T, c_uint) void;
const TypeMappingState = struct {
    sources: []mlir_capi.Type.T,
    target: mlir_capi.Type.T,
    identity_fallback: bool,
    recursive: bool,
    fn deinit(self: *@This()) void {
        std.heap.smp_allocator.free(self.sources);
        std.heap.smp_allocator.destroy(self);
    }
    fn conversion(type_: mlir_capi.Type.T, converted_type: [*c]mlir_capi.Type.T, user_data: ?*anyopaque) callconv(.c) c.MlirTypeConverterConversionStatus {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return c.MlirTypeConverterConversionStatusFailure));
        for (self.sources) |source| {
            if (c.mlirTypeEqual(type_, source)) {
                converted_type.* = self.target;
                return c.MlirTypeConverterConversionStatusSuccess;
            }
        }
        if (self.recursive and self.sources.len > 0) {
            const replaced = beaverConvertCompositeType(type_, self.sources.ptr, self.sources.len, self.target);
            if (!c.mlirTypeIsNull(replaced) and !c.mlirTypeEqual(replaced, type_)) {
                converted_type.* = replaced;
                return c.MlirTypeConverterConversionStatusSuccess;
            }
        }
        if (self.identity_fallback) {
            converted_type.* = type_;
            return c.MlirTypeConverterConversionStatusSuccess;
        }
        return c.MlirTypeConverterConversionStatusDeclined;
    }
};
fn TypeCallback(comptime name: []const u8, comptime Register: anytype, comptime callback: anytype) type {
    return struct {
        fn add(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const registration = try fetchTypeConverterRegistration(environment, args[0]);
            registration.mutex.lockUncancelable(std.Options.debug_io);
            defer registration.mutex.unlock(std.Options.debug_io);
            if (registration.closed.load(.acquire)) return error.TypeConverterClosed;
            const state = try appendCallback(TypeCallbackState, environment, &registration.callbacks, name, args[1], args[2]);
            Register(registration.handle, callback, state);
            return beam.make_ok(environment);
        }
    };
}
fn addTypeConversionMap(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const registration = try fetchTypeConverterRegistration(environment, args[0]);
    const io = std.Options.debug_io;
    registration.mutex.lockUncancelable(io);
    defer registration.mutex.unlock(io);
    if (registration.closed.load(.acquire)) return error.TypeConverterClosed;
    const sources = try callback_args.Range(mlir_capi.Type).decode(environment, args[1], std.heap.smp_allocator);
    errdefer std.heap.smp_allocator.free(sources);
    if (sources.len == 0) return error.EmptyTypeConversionMap;
    var identity_fallback = true;
    var recursive = false;
    var tup_n: c_int = 0;
    var tup_terms: [*c]const beam.term = undefined;
    if (e.enif_get_tuple(environment, args[3], &tup_n, &tup_terms) != 0 and tup_n == 2) {
        identity_fallback = try beam.get_bool(environment, tup_terms[0]);
        recursive = try beam.get_bool(environment, tup_terms[1]);
    } else {
        identity_fallback = try beam.get_bool(environment, args[3]);
    }
    const state = try std.heap.smp_allocator.create(TypeMappingState);
    errdefer std.heap.smp_allocator.destroy(state);
    state.* = .{
        .sources = sources,
        .target = try mlir_capi.Type.resource.fetch(environment, args[2]),
        .identity_fallback = identity_fallback,
        .recursive = recursive,
    };
    try registration.mappings.append(state);
    c.mlirTypeConverterAddConversion(registration.handle, TypeMappingState.conversion, state);
    return beam.make_atom(environment, "ok");
}
pub fn convertType(environment: beam.env, arguments: beam.term) !beam.term {
    const args = try @import("execution.zig").tuple(environment, arguments, 2);
    const registration = try fetchTypeConverterRegistration(environment, args[0]);
    const type_ = try mlir_capi.Type.resource.fetch(environment, args[1]);
    const io = std.Options.debug_io;
    registration.mutex.lockUncancelable(io);
    defer registration.mutex.unlock(io);
    if (registration.closed.load(.acquire)) return beam.make_nil(environment);
    const converted = c.mlirTypeConverterConvertType(registration.handle, type_);
    return if (c.mlirTypeIsNull(converted)) beam.make_nil(environment) else try mlir_capi.Type.resource.make_kind(environment, converted);
}
const ValueRanges = struct {
    sizes: []const isize,
    values: []const mlir_capi.Value.T,
    pub fn encode(environment: beam.env, self: @This()) !beam.term {
        const ranges = try beam.allocator.alloc(beam.term, self.sizes.len);
        defer beam.allocator.free(ranges);
        var offset: usize = 0;
        for (self.sizes, ranges) |size, *range| {
            const len: usize = @intCast(size);
            if (len > self.values.len - offset) return error.InvalidOperandRange;
            range.* = try Values.encode(environment, self.values[offset..][0..len]);
            offset += len;
        }
        return beam.make_term_list(environment, ranges);
    }
};
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
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return c.beaverLogicalResultFailure()));
        const response = callback_args.invoke(
            self.dispatcher,
            "conversion_pattern",
            .{ Timestamp, mlir_capi.Operation, Values, mlir_capi.ConversionPatternRewriter },
            .{ {}, operation, Values.slice(n_operands, operands), rewriter },
        ) catch return c.beaverLogicalResultFailure();
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
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return c.beaverLogicalResultFailure()));
        const response = callback_args.invoke(
            self.dispatcher,
            "conversion_pattern_1_to_n",
            .{ Timestamp, mlir_capi.Operation, ValueRanges, mlir_capi.ConversionPatternRewriter },
            .{ {}, operation, ValueRanges{ .sizes = if (n_ranges == 0) &.{} else range_sizes[0..@intCast(n_ranges)], .values = Values.slice(n_operands, operands) }, rewriter },
        ) catch return c.beaverLogicalResultFailure();
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
    const dispatcher = try PatternDispatcher.initWithOptions(try beam.self(environment), callback_args.fixedOptions(PatternDispatcher, try beam.get_u64(environment, args[7])));
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
        registration.handle,
        callbacks,
        state,
        0,
        null,
    );
    c.mlirRewritePatternSetAdd(set, c.mlirConversionPatternAsRewritePattern(pattern));
    pattern_user_owned = false;
    return beam.make_atom(environment, "ok");
}
fn populateStructuralConversions(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const target_reg = try fetchTargetRegistration(environment, args[0]);
    const patterns = try mlir_capi.RewritePatternSet.resource.fetch(environment, args[1]);
    const conv_reg = try fetchTypeConverterRegistration(environment, args[2]);
    const domain_mask = try beam.get(c_uint, environment, args[3]);
    const io = std.Options.debug_io;
    target_reg.mutex.lockUncancelable(io);
    defer target_reg.mutex.unlock(io);
    conv_reg.mutex.lockUncancelable(io);
    defer conv_reg.mutex.unlock(io);
    if (target_reg.closed.load(.acquire)) return error.ConversionTargetClosed;
    if (conv_reg.closed.load(.acquire)) return error.TypeConverterClosed;
    beaverPopulateStructuralTypeConversions(target_reg.handle, patterns, conv_reg.handle, domain_mask);
    return beam.make_atom(environment, "ok");
}
fn applyConversion(environment: beam.env, args: anytype) !beam.term {
    var report = string_ref.TextBuffer.init(beam.allocator);
    defer report.deinit();
    const result = c.beaverConversionRun(args.mode, args.operation, args.target, args.patterns, args.config, if (args.report) string_ref.collectStringRef else null, &report);
    if (args.report) return beam.make_slice(environment, report.items);
    return mlir_capi.LogicalResult.resource.make_kind(environment, result);
}
pub fn convert(environment: beam.env, context: mlir_capi.Context.T, arguments: beam.term, profile: bool) !beam.term {
    const args = try @import("execution.zig").tuple(environment, arguments, 8);
    const target = try fetchTargetRegistration(environment, args[1]);
    const operation = try mlir_capi.Operation.resource.fetch(environment, args[2]);
    const patterns = try mlir_capi.FrozenRewritePatternSet.resource.fetch(environment, args[3]);
    const owns_patterns = try beam.get_bool(environment, args[4]);
    defer if (owns_patterns) c.mlirFrozenRewritePatternSetDestroy(patterns);
    if (!c.mlirContextEqual(context, target.context) or
        !c.mlirContextEqual(context, c.mlirOperationGetContext(operation))) return error.ContextMismatch;
    const config = c.mlirConversionConfigCreate();
    defer c.mlirConversionConfigDestroy(config);
    const folding = try beam.get_i64(environment, args[5]);
    const materializations = try beam.get_i64(environment, args[6]);
    if (folding >= 0) c.mlirConversionConfigSetFoldingMode(config, @intCast(folding));
    if (materializations >= 0) c.mlirConversionConfigEnableBuildMaterializations(config, materializations != 0);
    const mode = try beam.get(c_uint, environment, args[0]);
    const report = try beam.get_bool(environment, args[7]);
    const started = if (profile) systemNs() else 0;
    const io = std.Options.debug_io;
    target.mutex.lockUncancelable(io);
    const wait = if (profile) systemNs() - started else 0;
    const result = if (target.closed.load(.acquire)) beam.make_nil(environment) else diagnostic.call_with_diagnostics(environment, context, applyConversion, .{ environment, .{
        .mode = mode,
        .report = report,
        .operation = operation,
        .target = target.handle,
        .patterns = patterns,
        .config = config,
    } }) catch beam.make_nil(environment);
    target.mutex.unlock(io);
    var costs = [_]beam.term{ beam.make_i64(environment, if (profile) systemNs() - started else 0), beam.make_i64(environment, wait) };
    var output = [_]beam.term{ result, if (profile) beam.make_tuple(environment, &costs) else beam.make_nil(environment) };
    return beam.make_tuple(environment, &output);
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
    prelude.beaverRawNIF(@This(), "type_converter_add_conversion_map", 4),
    prelude.beaverRawNIF(@This(), "type_converter_add_source_materialization", 3),
    prelude.beaverRawNIF(@This(), "type_converter_add_target_materialization", 3),
    prelude.beaverRawNIF(@This(), "type_converter_add_1_to_n_target_materialization", 3),
    prelude.beaverRawNIF(@This(), "type_converter_destroy", 1),
    prelude.beaverRawNIF(@This(), "type_converter_reply_callback", 4),
    prelude.beaverRawNIF(@This(), "type_converter_reply_types", 4),
    prelude.beaverRawNIF(@This(), "type_converter_reply_value", 3),
    prelude.beaverRawNIF(@This(), "type_converter_reply_values", 4),
    prelude.beaverRawNIF(@This(), "conversion_pattern_add", 8),
    prelude.beaverRawNIF(@This(), "conversion_populate_structural", 4),
    prelude.beaverRawNIFDirtyCPU(OperationInventory, "operation_inventory", 1),
    prelude.beaverRawNIF(ProfileMetrics, "process_cpu_time", 0),
    prelude.beaverRawNIF(ProfileMetrics, "peak_rss", 0),
};
pub const conversion_target_create = TargetRegistration.create;
pub const conversion_target_add_static = addStaticLegality;
pub const conversion_target_add_dynamic_op = Legality(.operation).add;
pub const conversion_target_add_dynamic_dialect = Legality(.dialect).add;
pub const conversion_target_mark_recursively_legal = Legality(.recursively_legal).add;
pub const conversion_target_mark_unknown_dynamic = Legality(.unknown).add;
pub const conversion_target_destroy = TargetRegistration.destroy;
pub const type_converter_create = TypeConverterRegistration.create;
pub const type_converter_add_conversion =
    TypeCallback("convert_type", c.mlirTypeConverterAddConversion, TypeCallbackState.conversion).add;
pub const type_converter_add_1_to_n_conversion =
    TypeCallback("convert_types", c.mlirTypeConverterAdd1ToNConversion, TypeCallbackState.conversion1ToN).add;
pub const type_converter_add_conversion_map = addTypeConversionMap;
pub const type_converter_add_source_materialization =
    TypeCallback("source_materialization", c.mlirTypeConverterAddSourceMaterialization, TypeCallbackState.sourceMaterialization).add;
pub const type_converter_add_target_materialization =
    TypeCallback("target_materialization", c.mlirTypeConverterAddTargetMaterialization, TypeCallbackState.targetMaterialization).add;
pub const type_converter_add_1_to_n_target_materialization =
    TypeCallback("target_materialization_1_to_n", c.mlirTypeConverterAdd1ToNTargetMaterialization, TypeCallbackState.targetMaterialization1ToN).add;
pub const type_converter_destroy = TypeConverterRegistration.destroy;
pub const type_converter_reply_callback = replyHandle(mlir_capi.Type, false).nif;
pub const type_converter_reply_types = replyHandle(mlir_capi.Type, true).nif;
pub const type_converter_reply_value = replyHandle(mlir_capi.Value, false).nif;
pub const type_converter_reply_values = replyHandle(mlir_capi.Value, true).nif;
pub const conversion_pattern_add = addConversionPattern;
pub const conversion_populate_structural = populateStructuralConversions;
