const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const m = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");
const tuple = @import("execution.zig").tuple;
const Range = @import("callback_args.zig").Range;

const SafepointCollector = struct {
    env: beam.env,
    slot_count: usize = 0,
    slot_types: beam.term = undefined,
    value_slots: beam.term = undefined,
    initialized: bool = false,
    entries: std.array_list.Managed(beam.term) = .init(beam.allocator),
    failed: bool = false,

    fn collect(
        total_slots: usize,
        slot_types_ptr: [*c]const m.Type.T,
        num_colored: usize,
        colored_values_ptr: [*c]const m.Value.T,
        value_slots_ptr: [*c]const usize,
        sp_index: usize,
        sp_op: m.Operation.T,
        num_live: usize,
        live_values_ptr: [*c]const m.Value.T,
        live_slots_ptr: [*c]const usize,
        user_data: ?*anyopaque,
    ) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        if (self.failed) return;
        self.append(.{
            .types = Range(m.Type).slice(@intCast(total_slots), slot_types_ptr),
            .colored = Range(m.Value).slice(@intCast(num_colored), colored_values_ptr),
            .slots = value_slots_ptr,
            .index = sp_index,
            .op = sp_op,
            .live = Range(m.Value).slice(@intCast(num_live), live_values_ptr),
            .live_slots = live_slots_ptr,
        }) catch {
            self.failed = true;
        };
    }

    fn append(self: *@This(), data: anytype) !void {
        const env = self.env;
        if (!self.initialized) {
            self.initialized = true;
            self.slot_count = data.types.len;
            self.slot_types = try Range(m.Type).encode(env, data.types);
            const pairs = try beam.allocator.alloc(beam.term, data.colored.len);
            defer beam.allocator.free(pairs);
            for (data.colored, pairs, 0..) |value, *pair, i| {
                var fields = [_]beam.term{ try m.Value.resource.make_kind(env, value), beam.make_u64(env, data.slots[i]) };
                pair.* = beam.make_tuple(env, &fields);
            }
            self.value_slots = beam.make_term_list(env, pairs);
        }
        if (data.op.ptr == null) return;
        const slots = try beam.allocator.alloc(beam.term, data.live.len);
        defer beam.allocator.free(slots);
        for (slots, 0..) |*slot, i| slot.* = beam.make_u64(env, data.live_slots[i]);
        var entry = [_]beam.term{
            beam.make_u64(env, data.index),
            try m.Operation.resource.make_kind(env, data.op),
            try Range(m.Value).encode(env, data.live),
            beam.make_term_list(env, slots),
        };
        try self.entries.append(beam.make_tuple(env, &entry));
    }

    fn finish(self: *@This()) !beam.term {
        if (self.failed or !self.initialized) return error.InvalidSafepointAnalysis;
        var out = [_]beam.term{
            beam.make_u64(self.env, self.slot_count),
            self.slot_types,
            self.value_slots,
            beam.make_term_list(self.env, self.entries.items),
        };
        return beam.make_tuple(self.env, &out);
    }
};

fn decodeSafepointOps(env: beam.env, list_term: beam.term) ![]m.Operation.T {
    return Range(m.Operation).decode(env, list_term, beam.allocator);
}

const Session = struct {
    handle: ?*anyopaque,
    context: m.Context.T,
    owner: beam.pid,
    mutation_depth: u32 = 0,
    mutex: @import("mutex.zig").Mutex = .{},

    fn releaseHandle(self: *@This()) void {
        if (self.handle) |handle| c.beaverAnalysisDestroy(handle);
        self.handle = null;
    }
    fn destroy(self: *@This()) void {
        self.mutex.lock();
        self.releaseHandle();
        self.mutex.unlock();
        self.mutex.destroy();
    }
    fn acquire(env: beam.env, ref: beam.term, caller: beam.pid) !*Session {
        const session = try Resource.fetch(env, ref);
        if (e.enif_compare_pids(&session.owner, &caller) != 0) return error.AnalysisOwnerMismatch;
        session.mutex.lock();
        return session;
    }
};
const Resource = kinda.RawResourceType(Session, "Beaver.MLIR.AnalysisSession", kinda.resourceDestructor(Session, Session.destroy));

pub fn analysis_create(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const root = try m.Operation.resource.fetch(env, args[0]);
    if (root.ptr == null) return error.NullAnalysisRoot;
    const owner = try beam.self(env);
    const handle = c.beaverAnalysisCreate(root);
    errdefer c.beaverAnalysisDestroy(handle);
    return Resource.make(env, .{ .handle = handle, .context = c.mlirOperationGetContext(root), .owner = owner });
}
pub fn analysis_close(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const session = try Session.acquire(env, args[0], try beam.self(env));
    defer session.mutex.unlock();
    if (session.mutation_depth != 0) return error.AnalysisMutationInProgress;
    session.releaseHandle();
    return beam.make_ok(env);
}
pub fn analysis_invalidate(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const session = try Session.acquire(env, args[0], try beam.self(env));
    defer session.mutex.unlock();
    const handle = session.handle orelse return error.ClosedAnalysisSession;
    c.beaverAnalysisInvalidate(handle, try beam.get(c_uint, env, args[1]));
    return beam.make_u64(env, c.beaverAnalysisGeneration(handle));
}

pub fn analysis_mutation(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const begin = try beam.get_bool(env, args[1]);
    const session = try Session.acquire(env, args[0], try beam.self(env));
    defer session.mutex.unlock();
    const handle = session.handle orelse return error.ClosedAnalysisSession;
    if (begin) {
        session.mutation_depth += 1;
    } else {
        if (session.mutation_depth == 0) return error.UnbalancedAnalysisMutation;
        session.mutation_depth -= 1;
    }
    c.beaverAnalysisInvalidate(handle, 0);
    return beam.make_ok(env);
}

pub fn query(env: beam.env, context: m.Context.T, args_term: beam.term, caller: beam.pid) !beam.term {
    const args = try tuple(env, args_term, 2);
    const session = try Session.acquire(env, args[0], caller);
    defer session.mutex.unlock();
    if (!c.mlirContextEqual(context, session.context)) return error.ContextMismatch;
    const handle = session.handle orelse return error.ClosedAnalysisSession;
    if (session.mutation_depth != 0) return error.AnalysisMutationInProgress;
    const count = try beam.get_list_length(env, args[1]);
    if (count > 256) return error.AnalysisBatchTooLarge;
    var rest = args[1];
    var results = std.array_list.Managed(beam.term).init(beam.allocator);
    defer results.deinit();
    for (0..count) |_| {
        const query_term = try beam.get_head_and_iter(env, &rest);
        const fields = try tuple(env, query_term, 3);
        const kind = try beam.get(c_uint, env, fields[0]);
        const op = try m.Operation.resource.fetch(env, fields[1]);
        if (kind == 3) {
            const type_ = try m.Type.resource.fetch(env, fields[2]);
            var layout: [9]u64 = undefined;
            if (!c.beaverAnalysisLayout(handle, op, type_, &layout)) return error.UnsupportedLayoutQuery;
            var terms: [9]beam.term = undefined;
            for (layout, &terms) |value, *term| term.* = beam.make_u64(env, value);
            try results.append(beam.make_tuple(env, &terms));
        } else if (kind == 4) {
            const sp_args = try tuple(env, fields[2], 2);
            try results.append(try analyzeSafepoints(env, handle, op, sp_args[0], try beam.get_bool(env, sp_args[1])));
        } else {
            const rhs = if (kind < 2) try m.Operation.resource.fetch(env, fields[2]) else m.Operation.T{ .ptr = null };
            const value = if (kind == 2) try m.Value.resource.fetch(env, fields[2]) else m.Value.T{ .ptr = null };
            const result = c.beaverAnalysisQuery(handle, kind, op, rhs, value);
            if (result < 0) return error.InvalidAnalysisQuery;
            try results.append(try beam.make(bool, env, result != 0));
        }
    }
    var output = [_]beam.term{ beam.make_u64(env, c.beaverAnalysisGeneration(handle)), beam.make_u64(env, c.beaverAnalysisCached(handle)), beam.make_term_list(env, results.items) };
    return beam.make_tuple(env, &output);
}
pub fn safepoint_analyze(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    return analyzeSafepoints(env, null, try m.Operation.resource.fetch(env, args[0]), args[1], try beam.get_bool(env, args[2]));
}

fn analyzeSafepoints(env: beam.env, session: ?*anyopaque, op: m.Operation.T, points: beam.term, exclude_entry: bool) !beam.term {
    const ops = try decodeSafepointOps(env, points);
    defer beam.allocator.free(ops);
    var collector = SafepointCollector{ .env = env };
    defer collector.entries.deinit();
    if (!c.beaverAnalyzeSafepoints(session, op, ops.ptr, ops.len, exclude_entry, SafepointCollector.collect, &collector))
        return error.InvalidSafepointAnalysis;
    return collector.finish();
}

pub fn safepoint_lower_resumable(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const func_op = try m.Operation.resource.fetch(env, args[0]);
    const sp_ops = try decodeSafepointOps(env, args[1]);
    defer beam.allocator.free(sp_ops);
    const save_op = try string_ref.get_binary_as_string_ref(env, args[2]);
    const load_op = try string_ref.get_binary_as_string_ref(env, args[3]);
    const tuple_mode = e.enif_is_tuple(env, args[4]) != 0;
    var indices = [_]isize{ -1, -1, 0 };
    if (tuple_mode) {
        const fields = try tuple(env, args[4], 3);
        for (&indices, fields) |*index, field| index.* = try beam.get(isize, env, field);
    } else indices[0] = try beam.get(isize, env, args[4]);
    var frame_bytes: u64 = 0;
    if (!c.beaverLowerResumableCFG(func_op, sp_ops.ptr, sp_ops.len, save_op, load_op, indices[0], indices[1], indices[2], &frame_bytes)) {
        return error.ResumableCFGLoweringFailed;
    }
    return if (tuple_mode) beam.make_ok_term(env, beam.make_u64(env, frame_bytes)) else beam.make_ok(env);
}

pub fn open(env: beam.env) void {
    Resource.open(env);
}
pub const nifs = .{
    prelude.beaverRawNIF(@This(), "analysis_create", 1),
    prelude.beaverRawNIFDirtyCPU(@This(), "analysis_close", 1),
    prelude.beaverRawNIFDirtyCPU(@This(), "analysis_invalidate", 2),
    prelude.beaverRawNIFDirtyCPU(@This(), "analysis_mutation", 2),
    prelude.beaverRawNIFDirtyCPU(@This(), "safepoint_analyze", 3),
    prelude.beaverRawNIFDirtyCPU(@This(), "safepoint_lower_resumable", 5),
};
