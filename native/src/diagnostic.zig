const beam = @import("beam");
const std = @import("std");
const io = std.io;
const stderr = io.getStdErr().writer();
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const MutexToken = @import("logical_mutex.zig").Token;
const result = @import("kinda").result;

const BeaverDiagnostic = struct {
    handler: ?beam.pid = null,
    const Error = error{
        EnvAllocFailure,
        MsgSendFailure,
    };
    pub fn sendDiagnostic(diagnostic: c.MlirDiagnostic, userData: ?*anyopaque) !mlir_capi.LogicalResult.T {
        const ud: ?*@This() = @ptrCast(@alignCast(userData));
        const h = ud.?.*.handler.?;
        const env = e.enif_alloc_env() orelse return Error.EnvAllocFailure;
        var tuple_slice: []beam.term = try beam.allocator.alloc(beam.term, 3);
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_atom(env, "diagnostic");
        tuple_slice[1] = try mlir_capi.Diagnostic.resource.make(env, diagnostic);
        var token = MutexToken{};
        tuple_slice[2] = try beam.make_ptr_resource_wrapped(env, &token);
        if (!beam.send(env, h, beam.make_tuple(env, tuple_slice))) {
            return Error.MsgSendFailure;
        }
        return token.wait_logical();
    }
    pub fn deleteUserData(userData: ?*anyopaque) callconv(.C) void {
        const ud: ?*@This() = @ptrCast(@alignCast(userData));
        beam.allocator.destroy(ud.?);
    }
    pub fn errorHandler(diagnostic: c.MlirDiagnostic, userData: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        return sendDiagnostic(diagnostic, userData) catch return c.mlirLogicalResultFailure();
    }
};

fn do_attach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    var userData: ?*BeaverDiagnostic = try beam.allocator.create(BeaverDiagnostic);
    userData.?.handler = beam.get_pid(env, args[1]) catch null;
    const id = c.mlirContextAttachDiagnosticHandler(try mlir_capi.Context.resource.fetch(env, args[0]), BeaverDiagnostic.errorHandler, userData, BeaverDiagnostic.deleteUserData);
    return try mlir_capi.DiagnosticHandlerID.resource.make(env, id);
}

pub const nifs = .{result.nif("beaver_raw_context_attach_diagnostic_handler", 2, do_attach).entry};
