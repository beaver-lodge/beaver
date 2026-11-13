const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const m = @import("mlir_capi.zig");
const diagnostic = @import("diagnostic.zig");
const batch = @import("ir_batch.zig");
fn clock() i64 {
    return @intCast(c.beaverExecutionClock());
}
fn atom(env: beam.env, term: beam.term, name: []const u8) bool {
    return e.enif_is_identical(term, beam.make_atom(env, name)) != 0;
}
pub fn tuple(env: beam.env, term: beam.term, count: usize) ![]const beam.term {
    var n: c_int = 0;
    var terms: [*c]const beam.term = undefined;
    if (e.enif_get_tuple(env, term, &n, &terms) == 0 or n != count) return error.InvalidExecutionArguments;
    return terms[0..count];
}
const Command = enum { batch_schema, transform, analysis, verify, pass, rewrite, snapshot, update, build, referenced_symbols, parse, create_pass, create_pattern, destroy, speculatability, memory_effects, replace_uses, conversion, convert_type };
const Work = struct {
    context: m.Context.T,
    command: Command,
    arguments: beam.term,
    profile: bool,
    env: beam.env,
    recipient: beam.pid = undefined,
    id: beam.term = undefined,
    submitted: i64 = 0,
    setup_ns: i64 = 0,
    fn init(env: beam.env, args: [*c]const beam.term) !Work {
        const command = inline for (@typeInfo(Command).@"enum".fields) |field| {
            if (atom(env, args[1], field.name)) break @field(Command, field.name);
        } else return error.InvalidExecutionCommand;
        return .{
            .context = try m.Context.resource.fetch(env, args[0]),
            .command = command,
            .arguments = args[2],
            .profile = atom(env, args[3], "true"),
            .env = env,
        };
    }
    fn operation(self: Work, term: beam.term) !m.Operation.T {
        const op = try m.Operation.resource.fetch(self.env, term);
        if (op.ptr == null or !c.mlirContextEqual(self.context, c.mlirOperationGetContext(op))) return error.ContextMismatch;
        return op;
    }
    fn invoke(self: Work, comptime name: []const u8, args: anytype) !beam.term {
        const binding = inline for (args) |arg| {
            if (@TypeOf(arg) == m.Operation.T) break c.beaverExecutionBindOperation(arg);
        } else null;
        defer c.beaverExecutionUnbindOperation(binding);
        const f = kinda.BangFunc(prelude.allKinds, c, name).wrap_ret_call;
        return diagnostic.call_with_diagnostics(self.env, self.context, f, .{ self.env, args });
    }
    fn execute(self: Work) !beam.term {
        const env = self.env;
        switch (self.command) {
            .batch_schema => return beam.make_slice(env, @import("batch_protocol.zig").schema.source),
            .transform => return @import("prepared_transform.zig").apply(env, self.context, self.arguments, self.recipient),
            .analysis => return @import("analysis.zig").query(env, self.context, self.arguments, self.recipient),
            .conversion => return @import("conversion.zig").convert(env, self.context, self.arguments, self.profile),
            .convert_type => return @import("conversion.zig").convertType(env, self.arguments),
            .replace_uses => return @import("value.zig").replace(env, self.context, self.arguments, self.recipient, self.id),
            .speculatability, .memory_effects => {
                const op = try self.operation((try tuple(env, self.arguments, 1))[0]);
                const binding = c.beaverExecutionBindOperation(op);
                defer c.beaverExecutionUnbindOperation(binding);
                if (self.command == .memory_effects) return @import("callback_bridge.zig").memoryEffects(env, op);
                return beam.make_c_int(env, @intCast(c.mlirConditionallySpeculatableOpInterfaceGetSpeculatability(op)));
            },
            .parse => {
                const args = try tuple(env, self.arguments, 1);
                const source = try m.StringRef.resource.fetch(env, args[0]);
                return self.invoke("beaverModuleCreateParseTracked", .{ self.context, source });
            },
            .create_pass => return @import("pass.zig").create(env, self.arguments),
            .create_pattern => return @import("rewrite_pattern.zig").create(env, self.context, self.arguments),
            .destroy => {
                const args = try tuple(env, self.arguments, 2);
                inline for (.{ "PassManager", "RewritePatternSet", "FrozenRewritePatternSet" }) |name| {
                    if (atom(env, args[0], name)) {
                        @field(c, "mlir" ++ name ++ "Destroy")(try @field(m, name).resource.fetch(env, args[1]));
                        return beam.make_ok(env);
                    }
                }
                return error.InvalidDestroyKind;
            },
            .verify => {
                const args = try tuple(env, self.arguments, 2);
                const op = try self.operation(args[0]);
                return self.invoke("beaverOperationVerify", .{ op, try beam.get_bool(env, args[1]) });
            },
            .pass => {
                const args = try tuple(env, self.arguments, 2);
                const pm = try m.PassManager.resource.fetch(env, args[0]);
                const op = try self.operation(args[1]);
                if (!c.mlirContextEqual(self.context, c.beaverPassManagerGetContext(pm))) return error.ContextMismatch;
                c.beaverPassManagerTrackExecution(pm);
                return self.invoke("mlirPassManagerRunOnOp", .{ pm, op });
            },
            .rewrite => {
                const args = try tuple(env, self.arguments, 3);
                const op = try self.operation(args[0]);
                const patterns = try m.FrozenRewritePatternSet.resource.fetch(env, args[1]);
                const config = try m.GreedyRewriteDriverConfig.resource.fetch(env, args[2]);
                return self.invoke("mlirApplyPatternsAndFoldGreedilyWithOp", .{ op, patterns, config });
            },
            .snapshot => return batch.snapshot(env, self.context, self.arguments),
            .update => return diagnostic.call_with_diagnostics(env, self.context, batch.update, .{ env, self.context, self.arguments }),
            .build => return diagnostic.call_with_diagnostics(env, self.context, batch.build, .{ env, self.context, self.arguments }),
            .referenced_symbols => return batch.referencedSymbols(env, self.context, self.arguments),
        }
    }
    fn result(self: Work) beam.term {
        const started = if (self.profile) clock() else 0;
        const measurement = if (self.profile) c.beaverExecutionMeasureBegin() else null;
        const outcome = if (self.execute()) |value| beam.make_ok_term(self.env, value) else |err| beam.make_error_atom(self.env, @errorName(err));
        var metrics = [_]u64{ 0, 0 };
        if (measurement) |ptr| c.beaverExecutionMeasureEnd(ptr, &metrics);
        const native_ns = if (self.profile) clock() - started else 0;
        const costs = if (self.profile) costs: {
            var cost = [_]beam.term{
                e.enif_make_int64(self.env, if (self.submitted == 0) 0 else started - self.submitted),
                e.enif_make_int64(self.env, native_ns),
                e.enif_make_uint64(self.env, metrics[0]),
                e.enif_make_uint64(self.env, metrics[1]),
                e.enif_make_int64(self.env, self.setup_ns),
                e.enif_make_int64(self.env, if (self.profile) clock() else 0),
            };
            break :costs beam.make_tuple(self.env, &cost);
        } else beam.make_nil(self.env);
        var reply = [_]beam.term{ outcome, costs };
        return beam.make_tuple(self.env, &reply);
    }
    fn run(data: ?*anyopaque) callconv(.c) void {
        const self: *Work = @ptrCast(@alignCast(data orelse return));
        defer beam.allocator.destroy(self);
        const output = e.enif_alloc_env() orelse @panic("failed to allocate execution result environment");
        defer e.enif_free_env(output);
        var reply = [_]beam.term{ e.enif_make_copy(output, self.id), e.enif_make_copy(output, self.result()) };
        // Release retained registrations before the caller may destroy its context.
        e.enif_free_env(self.env);
        _ = beam.send_advanced(null, self.recipient, output, beam.make_tuple(output, &reply));
    }
};
pub fn execution_worker(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    var work = try Work.init(env, args);
    const start = if (work.profile) clock() else 0;
    work.recipient = try beam.self(env);
    const id = e.enif_make_ref(env);
    const owned = e.enif_alloc_env() orelse return error.FailedToAllocateEnvironment;
    errdefer e.enif_free_env(owned);
    work.env = owned;
    work.arguments = e.enif_make_copy(owned, work.arguments);
    work.id = e.enif_make_copy(owned, id);
    work.submitted = if (work.profile) clock() else 0;
    work.setup_ns = work.submitted - start;
    const worker = try beam.allocator.create(Work);
    errdefer beam.allocator.destroy(worker);
    worker.* = work;
    if (!c.beaverContextAddWork(work.context, Work.run, worker)) {
        if (work.command != .convert_type) return error.ContextThreadSafeLockingDisabled;
        // Standalone type conversion also supports contexts without thread-safe locking.
        const thread = try @import("std").Thread.spawn(.{}, Work.run, .{worker});
        thread.detach();
    }
    return id;
}
pub fn execution_dirty(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const work = try Work.init(env, args);
    switch (work.command) {
        .verify, .snapshot, .update, .build, .referenced_symbols, .batch_schema => {},
        else => return error.DirtyExecutionRequiresCallbackFreeCommand,
    }
    if (c.beaverExecutionHasCallbacks(work.context)) return error.CallbackContextRequiresWorker;
    return work.result();
}
pub fn execution_auto(env: beam.env, n: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try m.Context.resource.fetch(env, args[0]);
    if (c.beaverExecutionHasCallbacks(ctx)) return execution_worker(env, n, args);
    return e.enif_schedule_nif(env, "beaver_verify", e.ERL_NIF_DIRTY_JOB_CPU_BOUND, prelude.beaverRawNIFDirtyCPU(@This(), "execution_dirty", 4).fptr, n, args);
}
pub fn executor_snapshot(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const ctx = try m.Context.resource.fetch(env, args[0]);
    var values: [7]u64 = undefined;
    c.beaverExecutorSnapshot(ctx, &values);
    var terms: [7]beam.term = undefined;
    for (values, 0..) |value, i| terms[i] = e.enif_make_uint64(env, value);
    return beam.make_tuple(env, &terms);
}
pub fn executor_release(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return beam.make_bool(env, c.beaverExecutorReleaseContext(try m.Context.resource.fetch(env, args[0])));
}
pub fn execution_build_mode(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    return beam.make_atom(env, @tagName(@import("builtin").mode));
}
pub fn execution_clock(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    return e.enif_make_uint64(env, c.beaverExecutionClock());
}
pub const nifs = .{
    prelude.beaverRawNIF(@This(), "execution_worker", 4),
    prelude.beaverRawNIFDirtyCPU(@This(), "execution_dirty", 4),
    prelude.beaverRawNIF(@This(), "execution_auto", 4),
    prelude.beaverRawNIF(@This(), "executor_snapshot", 1),
    prelude.beaverRawNIFDirtyIO(@This(), "executor_release", 1),
    prelude.beaverRawNIF(@This(), "execution_build_mode", 0),
    prelude.beaverRawNIF(@This(), "execution_clock", 0),
};
