const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const mlir = @import("mlir_capi.zig");

const Receiver = struct {
    env: beam.env,
    reference: beam.term,
    pid: e.ErlNifPid,

    fn destroy(data: ?*anyopaque) callconv(.c) void {
        const self: *Receiver = @ptrCast(@alignCast(data.?));
        e.enif_free_env(self.env);
        beam.allocator.destroy(self);
    }

    fn send(json: c.MlirStringRef, data: ?*anyopaque) callconv(.c) void {
        const self: *Receiver = @ptrCast(@alignCast(data.?));
        const env = e.enif_alloc_env() orelse return;
        defer e.enif_free_env(env);
        var terms = [_]beam.term{
            e.enif_make_atom(env, "mlir_optimization_remark"),
            e.enif_make_copy(env, self.reference),
            beam.make_slice(env, json.data[0..json.length]),
        };
        const message = beam.make_tuple(env, &terms);
        _ = e.enif_send(null, &self.pid, env, message);
    }
};

pub fn optimization_remarks_attach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try mlir.Context.resource.fetch(env, args[0]);
    const filters = try @import("string_ref.zig").get_binary_as_string_ref(env, args[2]);
    const final_policy = try beam.get(bool, env, args[3]);
    var pid: e.ErlNifPid = undefined;
    if (e.enif_is_ref(env, args[1]) == 0 or e.enif_self(env, &pid) == null) return error.InvalidRemarkReceiver;
    const owned_env = e.enif_alloc_env() orelse return error.OutOfMemory;
    const receiver = beam.allocator.create(Receiver) catch {
        e.enif_free_env(owned_env);
        return error.OutOfMemory;
    };
    receiver.* = .{ .env = owned_env, .reference = e.enif_make_copy(owned_env, args[1]), .pid = pid };
    return beam.make(bool, env, c.beaverContextEnableOptimizationRemarks(ctx, filters, final_policy, Receiver.send, receiver, Receiver.destroy));
}

pub const nifs = .{prelude.beaverRawNIF(@This(), "optimization_remarks_attach", 4)};
