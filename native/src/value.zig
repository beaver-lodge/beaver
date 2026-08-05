const mlir_capi = @import("mlir_capi.zig");
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
    from: mlir_capi.Value.T,
    replacement: mlir_capi.Value.T,
    callback_error: ?anyerror = null,

    fn deinit(self: *Self) void {
        self.dispatcher.deinit();
        beam.allocator.destroy(self);
    }

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
        const response = self.dispatcher.invoke("filter", message_env, .{operand}) catch |err| {
            self.callback_error = err;
            return false;
        };
        return response.success;
    }

    fn send_result(self: *Self) void {
        const message_env = e.enif_alloc_env() orelse @panic("failed to allocate callback result environment");
        defer e.enif_free_env(message_env);

        const message = if (self.callback_error) |err| blk: {
            var terms = [_]beam.term{
                beam.make_atom(message_env, "replace_uses_with_if_error"),
                self.dispatcher.copyId(message_env),
                beam.make_atom(message_env, @errorName(err)),
            };
            break :blk beam.make_tuple(message_env, &terms);
        } else blk: {
            var terms = [_]beam.term{
                beam.make_atom(message_env, "replace_uses_with_if_done"),
                self.dispatcher.copyId(message_env),
            };
            break :blk beam.make_tuple(message_env, &terms);
        };

        _ = beam.send_advanced(null, self.dispatcher.handler, message_env, message);
    }

    fn run(user_data: ?*anyopaque) callconv(.c) void {
        const self: *Self = @ptrCast(@alignCast(user_data orelse return));
        defer self.deinit();
        c.mlirValueReplaceUsesWithIf(self.from, self.replacement, filter, self);
        self.send_result();
    }

    pub fn value_replace_uses_with_if(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const self = try beam.allocator.create(Self);
        errdefer beam.allocator.destroy(self);

        const dispatcher = try RuntimeDispatcher.init(try beam.self(env));
        errdefer dispatcher.deinit();
        const from = try mlir_capi.Value.resource.fetch(env, args[0]);
        const replacement = try mlir_capi.Value.resource.fetch(env, args[1]);

        self.* = .{
            .dispatcher = dispatcher,
            .from = from,
            .replacement = replacement,
        };
        self.dispatcher.setCallback("filter", args[2]);

        const context = c.mlirValueGetContext(self.from);
        if (!c.mlirContextEqual(context, c.mlirValueGetContext(self.replacement))) {
            return error.ValuesFromDifferentContexts;
        }
        if (!c.beaverContextAddWork(context, run, self)) {
            return error.ContextMultithreadingDisabled;
        }

        var result = [_]beam.term{
            beam.make_atom(env, "async"),
            self.dispatcher.copyId(env),
        };
        return beam.make_tuple(env, &result);
    }
};

pub const nifs = .{
    prelude.beaverRawNIF(ConditionalUseReplacer, "value_replace_uses_with_if", 3),
};
