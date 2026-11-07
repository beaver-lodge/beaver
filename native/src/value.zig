const mlir_capi = @import("mlir_capi.zig");
const callback_args = @import("callback_args.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;

const callback_names = .{"filter"};
const RuntimeDispatcher = kinda.callback_runtime.Dispatcher(callback_names);

const ConditionalUseReplacer = struct {
    const Self = @This();

    dispatcher: *RuntimeDispatcher,
    callback_error: ?anyerror = null,

    fn filter(op_operand: mlir_capi.OpOperand.T, user_data: ?*anyopaque) callconv(.c) bool {
        const self: *Self = @ptrCast(@alignCast(user_data orelse return false));
        if (self.callback_error != null) return false;

        const message_env = e.enif_alloc_env() orelse {
            self.callback_error = error.FailedToAllocateEnvironment;
            return false;
        };
        const operand = mlir_capi.OpOperand.resource.make_kind(message_env, op_operand) catch |err| {
            e.enif_free_env(message_env);
            self.callback_error = err;
            return false;
        };
        const response = callback_args.invokeTerms(self.dispatcher, "filter", message_env, .{operand}) catch |err| {
            self.callback_error = err;
            return false;
        };
        return response.success;
    }
};

pub fn replace(env: beam.env, context: mlir_capi.Context.T, arguments: beam.term, recipient: beam.pid, id: beam.term) !beam.term {
    const args = try @import("execution.zig").tuple(env, arguments, 3);
    const from = try mlir_capi.Value.resource.fetch(env, args[0]);
    const replacement = try mlir_capi.Value.resource.fetch(env, args[1]);
    if (!c.mlirContextEqual(context, c.mlirValueGetContext(from)) or
        !c.mlirContextEqual(context, c.mlirValueGetContext(replacement))) return error.ValuesFromDifferentContexts;
    const dispatcher = try RuntimeDispatcher.init(recipient);
    defer dispatcher.deinit();
    var callback = [_]beam.term{ id, args[2] };
    dispatcher.setCallback("filter", beam.make_tuple(env, &callback));
    var replacer = ConditionalUseReplacer{ .dispatcher = dispatcher };
    c.mlirValueReplaceUsesWithIf(from, replacement, ConditionalUseReplacer.filter, &replacer);
    if (replacer.callback_error) |err| return err;
    return beam.make_ok(env);
}
