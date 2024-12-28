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
const StringRefCollector = @import("string_ref.zig").StringRefCollector;

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

// collect diagnostic as {severity, loc, message, num_notes}
const DiagnosticAggregator = struct {
    const Container = std.ArrayList(beam.term);
    env: beam.env,
    container: Container = undefined,
    pub fn collectDiagnostic(diagnostic: c.MlirDiagnostic, userData: ?*@This()) !mlir_capi.LogicalResult.T {
        const env = userData.?.env;
        var note_col = StringRefCollector.init(env);
        c.mlirDiagnosticPrint(diagnostic, StringRefCollector.append, @constCast(@ptrCast(@alignCast(&note_col))));
        const note = try note_col.collect_and_destroy();
        var entry_slice: []beam.term = try beam.allocator.alloc(beam.term, 4);
        entry_slice[0] = try beam.make(i64, env, c.mlirDiagnosticGetSeverity(diagnostic));
        entry_slice[1] = try mlir_capi.Location.resource.make(env, c.mlirDiagnosticGetLocation(diagnostic));
        entry_slice[2] = note;
        const num_notes: usize = @intCast(c.mlirDiagnosticGetNumNotes(diagnostic));
        entry_slice[3] = try beam.make(usize, env, num_notes);
        userData.?.container.append(beam.make_tuple(env, entry_slice)) catch unreachable;
        defer beam.allocator.free(entry_slice);
        if (num_notes > 0) {
            const nested: []c.MlirDiagnostic = try beam.allocator.alloc(c.MlirDiagnostic, num_notes);
            defer beam.allocator.free(nested);
            for (0..num_notes) |i| {
                const nested_diagnostic = c.mlirDiagnosticGetNote(diagnostic, @intCast(i));
                if (c.beaverLogicalResultIsFailure(try collectDiagnostic(nested_diagnostic, userData))) {
                    return c.mlirLogicalResultFailure();
                }
            }
        }
        return c.mlirLogicalResultSuccess();
    }
    pub fn errorHandler(diagnostic: c.MlirDiagnostic, userData: ?*anyopaque) callconv(.C) mlir_capi.LogicalResult.T {
        return collectDiagnostic(diagnostic, @ptrCast(@alignCast(userData))) catch return c.mlirLogicalResultFailure();
    }
    pub fn deleteUserData(userData: ?*anyopaque) callconv(.C) void {
        const ud: ?*@This() = @ptrCast(@alignCast(userData));
        beam.allocator.destroy(ud.?);
    }
    fn init(env: beam.env) !*@This() {
        var userData = try beam.allocator.create(DiagnosticAggregator);
        userData.env = env;
        userData.container = Container.init(beam.allocator);
        return userData;
    }
    fn collect_and_destroy(this: *@This()) !beam.term {
        defer this.container.deinit();
        return beam.make_term_list(this.env, this.container.items);
    }
};

pub fn WithDiagnosticsNIF(comptime Kinds: anytype, c_: anytype, comptime name: anytype) e.ErlNifFunc {
    const bang = kinda.BangFunc(Kinds, c_, name);
    const nifPrefix = "Elixir.Beaver.MLIR.CAPI.";
    const nifSuffix = "WithDiagnostics";
    const AttachAndRun = struct {
        fn with_diagnostics(env: beam.env, n: c_int, args: [*c]const beam.term) !beam.term {
            const userData = try DiagnosticAggregator.init(env);
            const ctx = try mlir_capi.Context.resource.fetch(env, args[0]);
            const id = c.mlirContextAttachDiagnosticHandler(ctx, DiagnosticAggregator.errorHandler, @ptrCast(@alignCast(userData)), DiagnosticAggregator.deleteUserData);
            defer c.mlirContextDetachDiagnosticHandler(ctx, id);
            var res_slice: []beam.term = try beam.allocator.alloc(beam.term, 2);
            res_slice[0] = try bang.nif(env, n - 1, args[1..]);
            res_slice[1] = try DiagnosticAggregator.collect_and_destroy(userData);
            defer beam.allocator.free(res_slice);
            return beam.make_tuple(env, res_slice);
        }
    };
    return result.nif(nifPrefix ++ name ++ nifSuffix, 1 + bang.arity, AttachAndRun.with_diagnostics).entry;
}

fn do_attach(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    var userData: ?*BeaverDiagnostic = try beam.allocator.create(BeaverDiagnostic);
    userData.?.handler = beam.get_pid(env, args[1]) catch null;
    const id = c.mlirContextAttachDiagnosticHandler(try mlir_capi.Context.resource.fetch(env, args[0]), BeaverDiagnostic.errorHandler, userData, BeaverDiagnostic.deleteUserData);
    return try mlir_capi.DiagnosticHandlerID.resource.make(env, id);
}

pub const nifs = .{ result.nif("beaver_raw_context_attach_diagnostic_handler", 2, do_attach).entry};
