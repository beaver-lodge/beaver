const beam = @import("beam");
const std = @import("std");
const io = std.io;
const stderr = io.getStdErr().writer();
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const result = @import("kinda").result;

const BeaverDiagnostic = struct {
    handler: ?beam.pid = null,
    fn print_fragment(str: anytype, userData: ?*anyopaque) void {
        const ud: ?*@This() = @ptrCast(@alignCast(userData));
        if (ud) |ptr| {
            if (ptr.*.handler) |h| {
                const env = e.enif_alloc_env() orelse unreachable;
                const msg = beam.make_slice(env, str);
                defer e.enif_clear_env(env);
                _ = beam.send(env, h, msg);
                return;
            }
        }
        stderr.print("{s}", .{str}) catch return;
    }
    pub fn printDiagnostic(str: mlir_capi.StringRef.T, userData: ?*anyopaque) callconv(.C) void {
        print_fragment(str.data[0..str.length], userData);
    }
    pub fn deleteUserData(userData: ?*anyopaque) callconv(.C) void {
        const ud: ?*@This() = @ptrCast(@alignCast(userData));
        if (ud) |ptr| {
            beam.allocator.destroy(ptr);
        }
    }
    pub fn errorHandler(diagnostic: c.MlirDiagnostic, userData: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        print_fragment("[Beaver] [Diagnostic] [", userData);
        const loc = c.mlirDiagnosticGetLocation(diagnostic);
        c.beaverLocationPrint(loc, printDiagnostic, userData);
        print_fragment("] ", userData);

        c.mlirDiagnosticPrint(diagnostic, printDiagnostic, userData);
        print_fragment("\n", userData);

        const num_note = c.mlirDiagnosticGetNumNotes(diagnostic);
        var i: isize = 0;
        while (i < num_note) {
            const note_d = c.mlirDiagnosticGetNote(diagnostic, i);
            c.mlirDiagnosticPrint(note_d, printDiagnostic, userData);
            i += 1;
        }
        print_fragment("\n", userData);
        return c.mlirLogicalResultSuccess();
    }
};

fn do_attach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    var userData: ?*BeaverDiagnostic = try beam.allocator.create(BeaverDiagnostic);
    userData.?.handler = beam.get_pid(env, args[1]) catch null;
    const id = c.mlirContextAttachDiagnosticHandler(try mlir_capi.Context.resource.fetch(env, args[0]), BeaverDiagnostic.errorHandler, userData, BeaverDiagnostic.deleteUserData);
    return try mlir_capi.DiagnosticHandlerID.resource.make(env, id);
}

fn get_diagnostic_string_callback(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    return try mlir_capi.StringCallback.resource.make(env, BeaverDiagnostic.printDiagnostic);
}

pub const print = BeaverDiagnostic.printDiagnostic;
pub const nifs = .{ result.nif("beaver_raw_context_attach_diagnostic_handler", 2, do_attach).entry, result.nif("beaver_raw_get_diagnostic_string_callback", 0, get_diagnostic_string_callback).entry };
