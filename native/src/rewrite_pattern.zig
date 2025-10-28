const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const debug_print = @import("std").debug.print;
const result = @import("kinda").result;
const diagnostic = @import("diagnostic.zig");
const string_ref = @import("string_ref.zig");

const CallbackDispatcher = struct {
    handler: beam.pid,
    env: beam.env,
    id: beam.term,
    callbacks: struct { construct: ?beam.term = null, destruct: ?beam.term = null, matchAndRewrite: ?beam.term = null },

    fn construct(userData: ?*anyopaque) callconv(.c) void {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb_and_consume_env(this, "construct", temp_env, .{}) catch unreachable;
        if (c.beaverLogicalResultIsFailure(res)) @panic("Fail to construct a rewrite pattern implemented in Elixir");
    }

    fn destruct(userData: ?*anyopaque) callconv(.c) void {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const res = forward_cb_and_consume_env(this, "destruct", temp_env, .{}) catch unreachable;
        if (c.beaverLogicalResultIsFailure(res)) @panic("Fail to destruct a rewrite pattern implemented in Elixir");
        e.enif_free_env(this.env);
        beam.allocator.destroy(this);
    }

    fn matchAndRewrite(pattern: mlir_capi.RewritePattern.T, op: mlir_capi.Operation.T, rewriter: mlir_capi.PatternRewriter.T, userData: ?*anyopaque) callconv(.c) mlir_capi.LogicalResult.T {
        const this: *@This() = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const pattern_ = mlir_capi.RewritePattern.resource.make_kind(temp_env, pattern) catch @panic("failed to make resource for RewritePattern");
        const op_ = mlir_capi.Operation.resource.make_kind(temp_env, op) catch @panic("failed to make resource for Operation");
        const rewriter_ = mlir_capi.PatternRewriter.resource.make_kind(temp_env, rewriter) catch @panic("failed to make resource for PatternRewriter");
        return forward_cb_and_consume_env(this, "matchAndRewrite", temp_env, .{ pattern_, op_, rewriter_ }) catch @panic("failed to forward matchAndRewrite callback");
    }

    const kConstruct = "construct";
    const kDestruct = "destruct";
    const kMatchAndRewrite = "matchAndRewrite";
    const callback_names = .{ kConstruct, kDestruct, kMatchAndRewrite };

    const Token = @import("logical_mutex.zig").Token;

    fn forward_cb_and_consume_env(this: *@This(), comptime callback: []const u8, temp_env: beam.env, args: anytype) !mlir_capi.LogicalResult.T {
        if (e.enif_thread_type() != e.ERL_NIF_THR_UNDEFINED) {
            @panic("External pattern must be run on non-scheduler thread to prevent deadlock. Callback: " ++ callback);
        }
        const cb = @field(this.*.callbacks, callback);
        if (cb == null) {
            std.log.warn("callback skipped: {s}", .{callback});
            e.enif_free_env(temp_env);
            return c.mlirLogicalResultSuccess();
        } else {
            var token = Token{};
            var buffer = std.array_list.Managed(beam.term).init(beam.allocator);
            defer buffer.deinit();
            try buffer.append(beam.make_atom(temp_env, callback));
            try buffer.append(try beam.make_ptr_resource_wrapped(temp_env, &token));
            try buffer.append(e.enif_make_copy(temp_env, cb.?));
            try buffer.append(e.enif_make_copy(temp_env, this.id));
            inline for (args) |arg| {
                try buffer.append(arg);
            }
            const msg = beam.make_tuple(temp_env, buffer.items);
            errdefer e.enif_free_env(temp_env);
            if (!beam.send_advanced(temp_env, this.*.handler, temp_env, msg)) {
                @panic("Fail to send message to pattern server");
            }
            const ret = token.wait_logical();
            // Transfer owner ship to last caller
            this.handler = ret.caller;
            return ret.result;
        }
    }

    fn init(handler: beam.pid) !*@This() {
        const this = try beam.allocator.create(@This());
        const this_env = e.enif_alloc_env() orelse @panic("fail to allocate env for callback dispatcher");
        this.* = @This(){ .handler = handler, .env = this_env, .id = e.enif_make_unique_integer(this_env, e.ERL_NIF_UNIQUE_POSITIVE), .callbacks = .{} };
        return this;
    }

    const PatternCreator = struct {
        pid: beam.pid,
        rootName: beam.term,
        benefit: c_uint,
        context: mlir_capi.Context.T,
        cb_map: std.StringHashMap(beam.term),

        fn init(env: beam.env, args: [*c]const beam.term) !*@This() {
            if (e.enif_thread_type() == e.ERL_NIF_THR_UNDEFINED) {
                @panic("Must be called from scheduler thread to get self pid");
            }
            const this = try beam.allocator.create(@This());
            errdefer beam.allocator.destroy(this);
            if (e.enif_self(env, &this.pid) == null) {
                return error.FailToGetSelfPid;
            }
            this.rootName = args[0];
            this.benefit = try beam.get(c_uint, env, args[1]);
            this.context = try mlir_capi.Context.resource.fetch(env, args[2]);
            this.cb_map = std.StringHashMap(beam.term).init(beam.allocator);
            inline for (callback_names, 3..) |f, i| {
                const cb = args[i];
                if (!beam.is_nil2(env, cb)) {
                    try this.cb_map.put(f, cb);
                } else {
                    @panic("nil callback found: " ++ f);
                }
            }
            return this;
        }

        fn deinit(this: *@This()) void {
            this.cb_map.deinit();
            beam.allocator.destroy(this);
        }
        const StructOfCallbacks = c.MlirRewritePatternCallbacks{
            .construct = CallbackDispatcher.construct,
            .destruct = CallbackDispatcher.destruct,
            .matchAndRewrite = CallbackDispatcher.matchAndRewrite,
        };
        fn create_and_send(worker: ?*anyopaque) callconv(.c) void {
            if (e.enif_thread_type() != e.ERL_NIF_THR_UNDEFINED) {
                @panic("Must create rewrite pattern on non-scheduler thread to prevent deadlock");
            }
            const this: *@This() = @ptrCast(@alignCast(worker));
            defer this.deinit();

            const callback_dispatcher: *CallbackDispatcher = CallbackDispatcher.init(this.pid) catch @panic("fail to allocate creator for rewrite pattern");
            inline for (callback_names) |f| {
                const cb = this.cb_map.get(f).?;
                if (!beam.is_nil2(callback_dispatcher.env, cb)) {
                    @field(callback_dispatcher.callbacks, f) = e.enif_make_copy(callback_dispatcher.env, cb);
                } else {
                    @field(callback_dispatcher.callbacks, f) = null;
                }
            }

            const nGeneratedNames = 0;
            const generatedNames: [*c]mlir_capi.StringRef.T = null;
            const temp_env = e.enif_alloc_env() orelse unreachable;
            const rootName = string_ref.get_binary_as_string_ref(temp_env, this.rootName) catch @panic("fail to get root name binary");
            const pattern: beam.term = @call(.auto, kinda.BangFunc(prelude.allKinds, c, "mlirOpRewritePatternCreate").wrap_ret_call, .{ temp_env, .{ rootName, this.benefit, this.context, StructOfCallbacks, callback_dispatcher, nGeneratedNames, generatedNames } }) catch @panic("fail to create rewrite pattern");

            if (!beam.send_advanced(null, this.pid, temp_env, pattern)) {
                @panic("fail to send rewrite pattern");
            }
        }
    };

    fn PatternSetDestroyAsyncFunctor(field_name: []const u8) type {
        const PatternSetType: type = @field(mlir_capi, field_name);
        return struct {
            pid: beam.pid = undefined,
            set: PatternSetType.T,
            const Self = @This();
            fn destroy(worker: ?*anyopaque) callconv(.c) void {
                if (e.enif_thread_type() != e.ERL_NIF_THR_UNDEFINED) {
                    @panic("Must destroy pattern on non-scheduler thread to prevent deadlock");
                }
                const this: *Self = @ptrCast(@alignCast(worker));
                @call(.auto, @field(c, "mlir" ++ field_name ++ "Destroy"), .{this.set});
                defer beam.allocator.destroy(this);
                const temp_env = e.enif_alloc_env() orelse unreachable;
                if (!beam.send_advanced(temp_env, this.pid, temp_env, beam.make_atom(temp_env, "destroy_done"))) {
                    @panic("Fail to send message of applying rewrite pattern set");
                }
            }
            pub fn destroy_nif(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
                const context = try mlir_capi.Context.resource.fetch(env, args[0]);
                const set = try PatternSetType.resource.fetch(env, args[1]);
                const destroyer = try beam.allocator.create(@This());
                destroyer.* = @This(){ .set = set };
                if (e.enif_self(env, &destroyer.pid) == null) {
                    return error.FailToGetSelfPid;
                }
                if (c.beaverContextAddWork(context, destroy, @ptrCast(@constCast(destroyer)))) {
                    return beam.make_atom(env, "async");
                } else {
                    defer beam.allocator.destroy(destroyer);
                    return error.FailToDestroyRewritePattern;
                }
            }
        };
    }
    fn PatternSetApplyAsyncFunctor(ContainerType: anytype) type {
        const PatternSetType: type = mlir_capi.FrozenRewritePatternSet;
        return struct {
            set: PatternSetType.T,
            ir: ContainerType.T,
            cfg: mlir_capi.GreedyRewriteDriverConfig.T,
            pid: beam.pid = undefined,
            const Self = @This();

            fn do(worker: ?*anyopaque) callconv(.c) void {
                if (e.enif_thread_type() != e.ERL_NIF_THR_UNDEFINED) {
                    @panic("Must apply pattern on non-scheduler thread to prevent deadlock");
                }
                const this: *Self = @ptrCast(@alignCast(worker));
                defer beam.allocator.destroy(this);

                const temp_env = e.enif_alloc_env() orelse unreachable;
                const ctx = switch (ContainerType) {
                    mlir_capi.Module => c.mlirModuleGetContext(this.ir),
                    mlir_capi.Operation => c.mlirOperationGetContext(this.ir),
                    else => @panic("unsupported container type"),
                };
                const loc = switch (ContainerType) {
                    mlir_capi.Module => c.mlirOperationGetLocation(c.mlirModuleGetOperation(this.ir)),
                    mlir_capi.Operation => c.mlirOperationGetLocation(this.ir),
                    else => @panic("unsupported container type"),
                };
                const apply_func = switch (ContainerType) {
                    mlir_capi.Module => kinda.BangFunc(prelude.allKinds, c, "mlirApplyPatternsAndFoldGreedily").wrap_ret_call,
                    mlir_capi.Operation => kinda.BangFunc(prelude.allKinds, c, "mlirApplyPatternsAndFoldGreedilyWithOp").wrap_ret_call,
                    else => @panic("unsupported container type"),
                };
                const args = .{ this.ir, this.set, this.cfg };
                const result_with_diagnostics = diagnostic.call_with_diagnostics(temp_env, ctx, apply_func, .{ temp_env, args }) catch |err| {
                    c.mlirEmitError(loc, @errorName(err));
                    return;
                };

                if (!beam.send_advanced(temp_env, this.pid, temp_env, result_with_diagnostics)) {
                    @panic("Fail to send message of applying rewrite pattern set");
                }
            }
            pub fn nif(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
                const context = try mlir_capi.Context.resource.fetch(env, args[0]);
                const ir = try ContainerType.resource.fetch(env, args[1]);
                const set = try PatternSetType.resource.fetch(env, args[2]);
                const cfg = try mlir_capi.GreedyRewriteDriverConfig.resource.fetch(env, args[3]);
                const doer = try beam.allocator.create(@This());
                doer.* = @This(){ .set = set, .ir = ir, .cfg = cfg };
                if (e.enif_self(env, &doer.pid) == null) {
                    return error.FailToGetSelfPid;
                }
                if (c.beaverContextAddWork(context, do, @ptrCast(@constCast(doer)))) {
                    return beam.make_atom(env, "async");
                } else {
                    defer beam.allocator.destroy(doer);
                    return error.FailToApplyRewritePatternSet;
                }
            }
        };
    }
    pub fn create_mlir_rewrite_pattern(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const creator = try PatternCreator.init(env, args);
        const context = creator.context;
        if (c.beaverContextAddWork(context, PatternCreator.create_and_send, @ptrCast(@constCast(creator)))) {
            return beam.make_atom(env, "async");
        } else {
            defer creator.deinit();
            return error.FailToCreateRewritePattern;
        }
    }
    pub const destroy_frozen_rewrite_pattern_set = PatternSetDestroyAsyncFunctor("FrozenRewritePatternSet").destroy_nif;
    pub const destroy_rewrite_pattern_set = PatternSetDestroyAsyncFunctor("RewritePatternSet").destroy_nif;
    pub const apply_rewrite_pattern_set_with_module = PatternSetApplyAsyncFunctor(mlir_capi.Module).nif;
    pub const apply_rewrite_pattern_set_with_op = PatternSetApplyAsyncFunctor(mlir_capi.Operation).nif;
};

pub const nifs = .{
    prelude.beaverRawNIF(CallbackDispatcher, "create_mlir_rewrite_pattern", 6),
    prelude.beaverRawNIF(CallbackDispatcher, "destroy_frozen_rewrite_pattern_set", 2),
    prelude.beaverRawNIF(CallbackDispatcher, "destroy_rewrite_pattern_set", 2),
    prelude.beaverRawNIF(CallbackDispatcher, "apply_rewrite_pattern_set_with_module", 4),
    prelude.beaverRawNIF(CallbackDispatcher, "apply_rewrite_pattern_set_with_op", 4),
};
