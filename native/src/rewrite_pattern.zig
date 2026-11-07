const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const diagnostic = @import("diagnostic.zig");
const string_ref = @import("string_ref.zig");
const callback_names = .{ "construct", "destruct", "matchAndRewrite" };
const RuntimeDispatcher = kinda.callback_runtime.Dispatcher(callback_names);
const callback_args = @import("callback_args.zig");

const PatternState = struct {
    dispatcher: *RuntimeDispatcher,
    operands: isize,
    results: isize,
    attributes: []mlir_capi.StringRef.T,

    fn init(env: beam.env, owner: beam.pid, filter: beam.term) !*@This() {
        const dispatcher = try RuntimeDispatcher.initWithOptions(owner, callback_args.fixedOptions(RuntimeDispatcher, 30_000));
        errdefer dispatcher.deinit();
        inline for (callback_names) |name| dispatcher.setCallback(name, beam.make_atom(env, name));
        var arity: c_int = 0;
        var terms: [*c]const beam.term = undefined;
        if (e.enif_get_tuple(env, filter, &arity, &terms) == 0 or arity != 3) return error.InvalidPatternFilter;
        const operands = try beam.get(isize, env, terms[0]);
        const results = try beam.get(isize, env, terms[1]);
        if (operands < -1 or results < -1) return error.InvalidPatternFilter;
        const list = e.enif_make_copy(dispatcher.env, terms[2]);
        const attributes = try beam.allocator.alloc(mlir_capi.StringRef.T, try beam.get_list_length(dispatcher.env, list));
        errdefer beam.allocator.free(attributes);
        var rest = list;
        for (attributes) |*attribute| {
            const term = try beam.get_head_and_iter(dispatcher.env, &rest);
            attribute.* = try string_ref.get_binary_as_string_ref(dispatcher.env, term);
        }
        const this = try beam.allocator.create(@This());
        this.* = .{
            .dispatcher = dispatcher,
            .operands = operands,
            .results = results,
            .attributes = attributes,
        };
        return this;
    }

    fn deinit(this: *@This()) void {
        beam.allocator.free(this.attributes);
        this.dispatcher.deinit();
        beam.allocator.destroy(this);
    }

    fn matches(this: *@This(), operation: mlir_capi.Operation.T) bool {
        if (this.operands >= 0 and c.mlirOperationGetNumOperands(operation) != this.operands) return false;
        if (this.results >= 0 and c.mlirOperationGetNumResults(operation) != this.results) return false;
        for (this.attributes) |name| {
            if (c.mlirOperationGetAttributeByName(operation, name).ptr == null) return false;
        }
        return true;
    }
};

const CallbackDispatcher = struct {
    fn construct(userData: ?*anyopaque) callconv(.c) void {
        const this: *PatternState = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const response = this.dispatcher.invoke("construct", temp_env, .{}) catch unreachable;
        if (!response.success) @panic("Fail to construct a rewrite pattern implemented in Elixir");
    }

    fn destruct(userData: ?*anyopaque) callconv(.c) void {
        const this: *PatternState = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const response = this.dispatcher.invoke("destruct", temp_env, .{}) catch unreachable;
        if (!response.success) @panic("Fail to destruct a rewrite pattern implemented in Elixir");
        this.deinit();
    }

    fn matchAndRewrite(pattern: mlir_capi.RewritePattern.T, op: mlir_capi.Operation.T, rewriter: mlir_capi.PatternRewriter.T, userData: ?*anyopaque) callconv(.c) mlir_capi.LogicalResult.T {
        const this: *PatternState = @ptrCast(@alignCast(userData));
        if (!this.matches(op)) return c.beaverLogicalResultFailure();
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const pattern_ = mlir_capi.RewritePattern.resource.make_kind(temp_env, pattern) catch @panic("failed to make resource for RewritePattern");
        const op_ = mlir_capi.Operation.resource.make_kind(temp_env, op) catch @panic("failed to make resource for Operation");
        const rewriter_ = mlir_capi.PatternRewriter.resource.make_kind(temp_env, rewriter) catch @panic("failed to make resource for PatternRewriter");
        const response = this.dispatcher.invoke("matchAndRewrite", temp_env, .{ pattern_, op_, rewriter_ }) catch @panic("failed to forward matchAndRewrite callback");
        return if (response.success) c.beaverLogicalResultSuccess() else c.beaverLogicalResultFailure();
    }

    const PatternCreator = struct {
        pid: beam.pid,
        env: beam.env,
        rootName: beam.term,
        benefit: c_uint,
        context: mlir_capi.Context.T,
        state: *PatternState,

        fn init(env: beam.env, args: [*c]const beam.term) !*@This() {
            if (e.enif_thread_type() == e.ERL_NIF_THR_UNDEFINED) {
                @panic("Must be called from scheduler thread to get self pid");
            }
            const this = try beam.allocator.create(@This());
            errdefer beam.allocator.destroy(this);
            var pid: beam.pid = undefined;
            if (e.enif_self(env, &pid) == null) return error.FailToGetSelfPid;
            const persistent_env = e.enif_alloc_env() orelse return error.FailToAllocateEnvironment;
            errdefer e.enif_free_env(persistent_env);
            var owner: beam.pid = undefined;
            if (e.enif_get_local_pid(env, args[3], &owner) == 0) return error.InvalidCallbackOwner;
            const state = try PatternState.init(env, owner, args[4]);
            errdefer state.deinit();
            this.* = .{
                .pid = pid,
                .env = persistent_env,
                .rootName = e.enif_make_copy(persistent_env, args[0]),
                .benefit = try beam.get(c_uint, env, args[1]),
                .context = try mlir_capi.Context.resource.fetch(env, args[2]),
                .state = state,
            };
            return this;
        }

        fn deinit(this: *@This()) void {
            e.enif_free_env(this.env);
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

            const nGeneratedNames = 0;
            const generatedNames: [*c]mlir_capi.StringRef.T = null;
            const temp_env = e.enif_alloc_env() orelse unreachable;
            const rootName = string_ref.get_binary_as_string_ref(this.env, this.rootName) catch @panic("fail to get root name binary");
            const pattern: beam.term = @call(.auto, kinda.BangFunc(prelude.allKinds, c, "mlirOpRewritePatternCreate").wrap_ret_call, .{ temp_env, .{ rootName, this.benefit, this.context, StructOfCallbacks, this.state, nGeneratedNames, generatedNames } }) catch @panic("fail to create rewrite pattern");

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
                if (e.enif_self(env, &destroyer.pid) == null) return error.FailToGetSelfPid;
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
                if (e.enif_self(env, &doer.pid) == null) return error.FailToGetSelfPid;
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
            creator.state.deinit();
            return error.ContextThreadSafeLockingDisabled;
        }
    }
    pub const destroy_frozen_rewrite_pattern_set = PatternSetDestroyAsyncFunctor("FrozenRewritePatternSet").destroy_nif;
    pub const destroy_rewrite_pattern_set = PatternSetDestroyAsyncFunctor("RewritePatternSet").destroy_nif;
    pub const apply_rewrite_pattern_set_with_module = PatternSetApplyAsyncFunctor(mlir_capi.Module).nif;
    pub const apply_rewrite_pattern_set_with_op = PatternSetApplyAsyncFunctor(mlir_capi.Operation).nif;
};

pub const nifs = .{
    prelude.beaverRawNIF(CallbackDispatcher, "create_mlir_rewrite_pattern", 5),
    prelude.beaverRawNIF(CallbackDispatcher, "destroy_frozen_rewrite_pattern_set", 2),
    prelude.beaverRawNIF(CallbackDispatcher, "destroy_rewrite_pattern_set", 2),
    prelude.beaverRawNIF(CallbackDispatcher, "apply_rewrite_pattern_set_with_module", 4),
    prelude.beaverRawNIF(CallbackDispatcher, "apply_rewrite_pattern_set_with_op", 4),
};
