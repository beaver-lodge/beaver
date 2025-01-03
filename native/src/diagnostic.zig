const beam = @import("beam");
const std = @import("std");
const io = std.io;
const stderr = io.getStdErr().writer();
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const MutexToken = @import("logical_mutex.zig").Token;
const kinda = @import("kinda");
const result = @import("kinda").result;
const PrinterNIF = @import("string_ref.zig").PrinterNIF;

// collect diagnostic as {severity, loc, message, nested_notes}
const DiagnosticAggregator = struct {
    const Container = std.ArrayList(beam.term);
    env: beam.env,
    container: Container = undefined,
    fn collectDiagnosticNested(diagnostic: c.MlirDiagnostic, userData: ?*@This()) !beam.term {
        const env = userData.?.env;
        const severity = c.mlirDiagnosticGetSeverity(diagnostic);
        const location = try mlir_capi.Location.resource.make(env, c.mlirDiagnosticGetLocation(diagnostic));
        const note = try PrinterNIF(mlir_capi.Diagnostic, c.mlirDiagnosticPrint).print_make(env, diagnostic);

        const num_notes: usize = @intCast(c.mlirDiagnosticGetNumNotes(diagnostic));
        var nested_notes: []beam.term = try beam.allocator.alloc(beam.term, num_notes);
        defer beam.allocator.free(nested_notes);
        for (0..num_notes) |i| {
            const nested_diagnostic = c.mlirDiagnosticGetNote(diagnostic, @intCast(i));
            nested_notes[i] = try collectDiagnosticNested(nested_diagnostic, userData);
        }

        var entry_slice: []beam.term = try beam.allocator.alloc(beam.term, 4);
        defer beam.allocator.free(entry_slice);
        entry_slice[0] = try beam.make(i64, env, severity);
        entry_slice[1] = location;
        entry_slice[2] = note;
        entry_slice[3] = beam.make_term_list(env, nested_notes);
        return beam.make_tuple(env, entry_slice);
    }

    fn collectDiagnosticTopLevel(diagnostic: c.MlirDiagnostic, userData: ?*@This()) !mlir_capi.LogicalResult.T {
        try userData.?.container.append(try collectDiagnosticNested(diagnostic, userData));
        return c.mlirLogicalResultSuccess();
    }
    fn errorHandler(diagnostic: c.MlirDiagnostic, userData: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        return collectDiagnosticTopLevel(diagnostic, @ptrCast(@alignCast(userData))) catch return c.mlirLogicalResultFailure();
    }
    fn deleteUserData(userData: ?*anyopaque) callconv(.C) void {
        const this: ?*@This() = @ptrCast(@alignCast(userData));
        this.?.container.deinit();
        beam.allocator.destroy(this.?);
    }
    fn init(env: beam.env) !*@This() {
        var userData = try beam.allocator.create(DiagnosticAggregator);
        userData.env = env;
        userData.container = Container.init(beam.allocator);
        return userData;
    }
    fn to_list(this: *@This()) !beam.term {
        return beam.make_term_list(this.env, this.container.items);
    }
};

pub fn call_with_diagnostics(env: beam.env, ctx: mlir_capi.Context.T, f: anytype, args: anytype) !beam.term {
    const userData = try DiagnosticAggregator.init(env);
    const id = c.mlirContextAttachDiagnosticHandler(ctx, DiagnosticAggregator.errorHandler, @ptrCast(@alignCast(userData)), DiagnosticAggregator.deleteUserData);
    defer c.mlirContextDetachDiagnosticHandler(ctx, id);
    var res_slice: []beam.term = try beam.allocator.alloc(beam.term, 2);
    defer beam.allocator.free(res_slice);
    res_slice[0] = try @call(.auto, f, args);
    res_slice[1] = try DiagnosticAggregator.to_list(userData);
    return beam.make_tuple(env, res_slice);
}

pub fn WithDiagnosticsNIF(comptime Kinds: anytype, c_: anytype, comptime name: anytype) e.ErlNifFunc {
    const bang = kinda.BangFunc(Kinds, c_, name);
    const nifPrefix = "Elixir.Beaver.MLIR.CAPI.";
    const nifSuffix = "WithDiagnostics";
    const AttachAndRun = struct {
        fn with_diagnostics(env: beam.env, n: c_int, args: [*c]const beam.term) !beam.term {
            const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
            return call_with_diagnostics(env, ctx, bang.nif, .{ env, n - 1, args[1..] });
        }
    };
    return result.nif(nifPrefix ++ name ++ nifSuffix, 1 + bang.arity, AttachAndRun.with_diagnostics).entry;
}

pub const nifs = .{};
