const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;

pub fn fixedOptions(comptime Dispatcher: type, timeout_ms: ?u64) Dispatcher.Options {
    var options: Dispatcher.Options = .{ .timeout_ms = timeout_ms };
    // Older published Kinda versions retain the serialized bridge. The pinned
    // runtime can release its dispatcher lock before waiting for a reply.
    if (@hasField(Dispatcher.Options, "ownership")) options.ownership = .fixed;
    return options;
}

/// Encodes context-owned handles in a fresh message environment, then transfers
/// that environment to the dispatcher. Failed encodings release it here.
pub fn invoke(dispatcher: anytype, comptime name: []const u8, comptime Kinds: anytype, values: anytype) !kinda.callback_runtime.Response {
    const c = @import("prelude.zig").c;
    const scope = inline for (Kinds, values) |Kind, value| {
        if (Kind == @import("mlir_capi.zig").Operation) break c.beaverExecutionEnterOperation(value);
    } else null;
    defer c.beaverExecutionLeaveOperation(scope);
    const environment = e.enif_alloc_env() orelse return error.FailedToAllocateEnvironment;
    var terms: @Tuple(&([_]type{beam.term} ** Kinds.len)) = undefined;
    inline for (Kinds, values, 0..) |Kind, value, index| {
        const encoded = if (@hasDecl(Kind, "encode")) Kind.encode(environment, value) else kinda.callback_adapter.handle(Kind, environment, value);
        terms[index] = encoded catch |err| {
            e.enif_free_env(environment);
            return err;
        };
    }
    return invokeTerms(dispatcher, name, environment, terms);
}

pub fn invokeTerms(dispatcher: anytype, comptime name: []const u8, environment: beam.env, terms: anytype) !kinda.callback_runtime.Response {
    const c = @import("prelude.zig").c;
    const start = c.beaverExecutionCallbackBegin();
    defer c.beaverExecutionCallbackEnd(start);
    return dispatcher.invoke(name, environment, terms);
}

pub fn Range(comptime Kind: type) type {
    return struct {
        pub fn encode(environment: beam.env, values: []const Kind.T) !beam.term {
            return kinda.callback_adapter.handleRange(Kind, environment, values);
        }

        pub fn slice(count: isize, values: [*c]const Kind.T) []const Kind.T {
            return if (count == 0) &.{} else values[0..@intCast(count)];
        }
    };
}

pub fn Optional(comptime Kind: type) type {
    return struct {
        pub fn encode(environment: beam.env, value: Kind.T) !beam.term {
            return if (value.ptr == null) beam.make_nil(environment) else kinda.callback_adapter.handle(Kind, environment, value);
        }
    };
}
