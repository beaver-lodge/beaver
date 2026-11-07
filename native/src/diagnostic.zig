const std = @import("std");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig").c;
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const result = @import("kinda").result;
const mutex = @import("mutex.zig");
const PrinterNIF = @import("string_ref.zig").PrinterNIF;

// collect diagnostic as {severity, loc, message, nested_notes}
const DiagnosticAggregator = struct {
    const Container = std.array_list.Managed(beam.term);
    env: beam.env,
    container: Container = undefined,
    mutex: mutex.Mutex = .{},
    fn collectDiagnosticNested(self: *@This(), diagnostic: c.MlirDiagnostic) !beam.term {
        const env = self.env;
        const severity = c.mlirDiagnosticGetSeverity(diagnostic);
        const location = try mlir_capi.Location.resource.make(env, c.mlirDiagnosticGetLocation(diagnostic));
        const note = try PrinterNIF(mlir_capi.Diagnostic, c.mlirDiagnosticPrint).print_make(env, diagnostic);

        const num_notes: usize = @intCast(c.mlirDiagnosticGetNumNotes(diagnostic));
        var nested_notes: []beam.term = try beam.allocator.alloc(beam.term, num_notes);
        defer beam.allocator.free(nested_notes);
        for (0..num_notes) |i| {
            const nested_diagnostic = c.mlirDiagnosticGetNote(diagnostic, @intCast(i));
            nested_notes[i] = try self.collectDiagnosticNested(nested_diagnostic);
        }
        var entry = [_]beam.term{ try beam.make(i64, env, severity), location, note, beam.make_term_list(env, nested_notes) };
        return beam.make_tuple(env, &entry);
    }
    fn errorHandler(diagnostic: c.MlirDiagnostic, userData: ?*anyopaque) callconv(.c) mlir_capi.LogicalResult.T {
        const self: *@This() = @ptrCast(@alignCast(userData orelse return c.beaverLogicalResultFailure()));
        self.mutex.lock();
        defer self.mutex.unlock();
        const entry = self.collectDiagnosticNested(diagnostic) catch return c.beaverLogicalResultFailure();
        self.container.append(entry) catch return c.beaverLogicalResultFailure();
        return c.beaverLogicalResultSuccess();
    }
    fn to_list(this: *@This(), destination_env: beam.env) !beam.term {
        this.mutex.lock();
        defer this.mutex.unlock();

        return e.enif_make_copy(destination_env, beam.make_term_list(this.env, this.container.items));
    }
};

pub fn call_with_diagnostics(env: beam.env, ctx: mlir_capi.Context.T, f: anytype, args: anytype) !beam.term {
    const diagnostic_env = e.enif_alloc_env() orelse return error.FailedToAllocateDiagnosticEnvironment;
    defer e.enif_free_env(diagnostic_env);
    var collector = DiagnosticAggregator{ .env = diagnostic_env, .container = .init(beam.allocator) };
    defer collector.container.deinit();
    defer collector.mutex.destroy();
    // Detach synchronizes with emission before the scoped collector is released.
    const id = c.mlirContextAttachDiagnosticHandler(ctx, DiagnosticAggregator.errorHandler, &collector, null);
    defer c.mlirContextDetachDiagnosticHandler(ctx, id);
    var reply = [_]beam.term{ try @call(.auto, f, args), try collector.to_list(env) };
    return beam.make_tuple(env, &reply);
}

pub fn WithDiagnosticsNIF(comptime name: anytype) e.ErlNifFunc {
    const bang = kinda.BangFunc(@import("prelude.zig").allKinds, c, name);
    const nifSuffix = "WithDiagnostics";
    const AttachAndRun = struct {
        fn with_diagnostics(env: beam.env, n: c_int, args: [*c]const beam.term) !beam.term {
            const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
            return call_with_diagnostics(env, ctx, bang.nif, .{ env, n - 1, args[1..] });
        }
    };
    return result.nif_with_flags(
        name ++ nifSuffix,
        1 + bang.arity,
        AttachAndRun.with_diagnostics,
        e.ERL_NIF_DIRTY_JOB_CPU_BOUND,
    ).entry;
}

pub const nifs = .{};
