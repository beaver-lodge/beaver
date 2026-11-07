const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
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
        const response = callback_args.invokeTerms(this.dispatcher, "construct", temp_env, .{}) catch unreachable;
        if (!response.success) @panic("Fail to construct a rewrite pattern implemented in Elixir");
    }

    fn destruct(userData: ?*anyopaque) callconv(.c) void {
        const this: *PatternState = @ptrCast(@alignCast(userData));
        const temp_env = e.enif_alloc_env() orelse unreachable;
        const response = callback_args.invokeTerms(this.dispatcher, "destruct", temp_env, .{}) catch unreachable;
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
        const response = callback_args.invokeTerms(this.dispatcher, "matchAndRewrite", temp_env, .{ pattern_, op_, rewriter_ }) catch @panic("failed to forward matchAndRewrite callback");
        return if (response.success) c.beaverLogicalResultSuccess() else c.beaverLogicalResultFailure();
    }
};

pub fn create(env: beam.env, ctx: mlir_capi.Context.T, arguments: beam.term) !beam.term {
    const args = try @import("execution.zig").tuple(env, arguments, 4);
    const root = try string_ref.get_binary_as_string_ref(env, args[0]);
    const benefit = try beam.get(c_uint, env, args[1]);
    var owner: beam.pid = undefined;
    if (e.enif_get_local_pid(env, args[2], &owner) == 0) return error.InvalidCallbackOwner;
    const state = try PatternState.init(env, owner, args[3]);
    errdefer state.deinit();
    const callbacks = c.MlirRewritePatternCallbacks{
        .construct = CallbackDispatcher.construct,
        .destruct = CallbackDispatcher.destruct,
        .matchAndRewrite = CallbackDispatcher.matchAndRewrite,
    };
    return kinda.BangFunc(prelude.allKinds, c, "mlirOpRewritePatternCreate").wrap_ret_call(env, .{ root, benefit, ctx, callbacks, state, 0, null });
}
