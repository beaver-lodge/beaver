const beam = @import("beam");
const std = @import("std");
const io = std.io;
const stderr = io.getStdErr().writer();
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const result = @import("result.zig");

const BeaverDiagnostic = struct {
    handler: beam.pid,
    fn printToMsg(str: anytype, userData: *@This()) void {
        const env = e.enif_alloc_env() orelse unreachable;
        defer e.enif_clear_env(env);
        const msg = beam.make_slice(env, str);
        _ = beam.send(env, userData.*.handler, msg);
    }
    fn printSlice(str: anytype, userData: ?*anyopaque) void {
        const ud: ?*@This() = @ptrCast(@alignCast(userData));
        if (ud) |ptr| {
            printToMsg(str, ptr);
        } else {
            stderr.print("{s}", .{str}) catch return;
        }
    }
    pub fn printDiagnostic(str: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.C) void {
        printSlice(str.data[0..str.length], userData);
    }
    pub fn deleteUserData(userData: ?*anyopaque) callconv(.C) void {
        const ud: ?*@This() = @ptrCast(@alignCast(userData));
        if (ud) |ptr| {
            beam.allocator.destroy(ptr);
        }
    }
    pub fn errorHandler(diagnostic: c.MlirDiagnostic, userData: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        printSlice("[Beaver] [Diagnostic] [", userData);
        const loc = c.mlirDiagnosticGetLocation(diagnostic);
        c.beaverLocationPrint(loc, printDiagnostic, userData);
        printSlice("] ", userData);

        c.mlirDiagnosticPrint(diagnostic, printDiagnostic, userData);
        printSlice("\n", userData);

        const num_note = c.mlirDiagnosticGetNumNotes(diagnostic);
        var i: isize = 0;
        while (i < num_note) {
            const note_d = c.mlirDiagnosticGetNote(diagnostic, i);
            c.mlirDiagnosticPrint(note_d, printDiagnostic, userData);
            i += 1;
        }
        printSlice("\n", userData);
        return c.mlirLogicalResultSuccess();
    }
};

fn do_attach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    var arg0: mlir_capi.Context.T = try mlir_capi.Context.resource.fetch(env, args[0]);
    var handler: ?beam.pid = undefined;
    handler = beam.get_pid(env, args[1]) catch null;
    var userData: ?*BeaverDiagnostic = null;
    if (handler) |h| {
        userData = try beam.allocator.create(BeaverDiagnostic);
        userData.?.handler = h;
    }
    const id = c.mlirContextAttachDiagnosticHandler(arg0, BeaverDiagnostic.errorHandler, userData, BeaverDiagnostic.deleteUserData);
    return try mlir_capi.DiagnosticHandlerID.resource.make(env, id);
}

pub const print = BeaverDiagnostic.printDiagnostic;
pub const attach = result.nif("beaver_raw_context_attach_diagnostic_handler", 2, do_attach).entry;
