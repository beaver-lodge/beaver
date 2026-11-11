const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const m = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");
const tuple = @import("execution.zig").tuple;

const SafepointCallback = *const fn (
    usize,
    [*]const m.Type.T,
    usize,
    [*]const m.Value.T,
    [*]const usize,
    usize,
    m.Operation.T,
    usize,
    [*]const m.Value.T,
    [*]const usize,
    ?*anyopaque,
) callconv(.c) void;

extern fn beaverAnalyzeSafepoints(
    ?*anyopaque,
    m.Operation.T,
    [*]const m.Operation.T,
    usize,
    bool,
    SafepointCallback,
    ?*anyopaque,
) bool;

extern fn beaverLowerResumableCFG(
    m.Operation.T,
    [*]const m.Operation.T,
    usize,
    m.StringRef.T,
    m.StringRef.T,
    isize,
    isize,
    isize,
    *u64,
) bool;

const SafepointCollector = struct {
    env: beam.env,
    slot_count: usize = 0,
    slot_types: beam.term = undefined,
    value_slots: beam.term = undefined,
    initialized: bool = false,
    entries: std.array_list.Managed(beam.term),
    failed: bool = false,

    fn init(env: beam.env) @This() {
        return .{
            .env = env,
            .entries = std.array_list.Managed(beam.term).init(beam.allocator),
        };
    }

    fn deinit(self: *@This()) void {
        self.entries.deinit();
    }

    fn collect(
        total_slots: usize,
        slot_types_ptr: [*]const m.Type.T,
        num_colored: usize,
        colored_values_ptr: [*]const m.Value.T,
        value_slots_ptr: [*]const usize,
        sp_index: usize,
        sp_op: m.Operation.T,
        num_live: usize,
        live_values_ptr: [*]const m.Value.T,
        live_slots_ptr: [*]const usize,
        user_data: ?*anyopaque,
    ) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        if (self.failed) return;
        const env = self.env;

        if (!self.initialized) {
            self.initialized = true;
            self.slot_count = total_slots;
            const st_terms = beam.allocator.alloc(beam.term, total_slots) catch {
                self.failed = true;
                return;
            };
            defer beam.allocator.free(st_terms);
            for (0..total_slots) |i| {
                st_terms[i] = m.Type.resource.make_kind(env, slot_types_ptr[i]) catch {
                    self.failed = true;
                    return;
                };
            }
            self.slot_types = beam.make_term_list(env, st_terms);

            const vs_terms = beam.allocator.alloc(beam.term, num_colored) catch {
                self.failed = true;
                return;
            };
            defer beam.allocator.free(vs_terms);
            for (0..num_colored) |i| {
                const v_term = m.Value.resource.make_kind(env, colored_values_ptr[i]) catch {
                    self.failed = true;
                    return;
                };
                var pair = [_]beam.term{ v_term, beam.make_u64(env, value_slots_ptr[i]) };
                vs_terms[i] = beam.make_tuple(env, &pair);
            }
            self.value_slots = beam.make_term_list(env, vs_terms);
        }

        if (sp_op.ptr == null) return;

        const lv_terms = beam.allocator.alloc(beam.term, num_live) catch {
            self.failed = true;
            return;
        };
        defer beam.allocator.free(lv_terms);
        const ls_terms = beam.allocator.alloc(beam.term, num_live) catch {
            self.failed = true;
            return;
        };
        defer beam.allocator.free(ls_terms);
        for (0..num_live) |i| {
            lv_terms[i] = m.Value.resource.make_kind(env, live_values_ptr[i]) catch {
                self.failed = true;
                return;
            };
            ls_terms[i] = beam.make_u64(env, live_slots_ptr[i]);
        }
        const op_term = m.Operation.resource.make_kind(env, sp_op) catch {
            self.failed = true;
            return;
        };
        var entry = [_]beam.term{
            beam.make_u64(env, sp_index),
            op_term,
            beam.make_term_list(env, lv_terms),
            beam.make_term_list(env, ls_terms),
        };
        self.entries.append(beam.make_tuple(env, &entry)) catch {
            self.failed = true;
            return;
        };
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
    const len = try beam.get_list_length(env, list_term);
    const ops = try beam.allocator.alloc(m.Operation.T, len);
    errdefer beam.allocator.free(ops);
    var rest = list_term;
    for (ops) |*item| {
        item.* = try m.Operation.resource.fetch(env, try beam.get_head_and_iter(env, &rest));
    }
    return ops;
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
            const sp_ops = try decodeSafepointOps(env, sp_args[0]);
            defer beam.allocator.free(sp_ops);
            const exclude_entry_args = try beam.get_bool(env, sp_args[1]);
            var collector = SafepointCollector.init(env);
            defer collector.deinit();
            if (!beaverAnalyzeSafepoints(handle, op, sp_ops.ptr, sp_ops.len, exclude_entry_args, SafepointCollector.collect, &collector)) {
                return error.InvalidSafepointAnalysis;
            }
            try results.append(try collector.finish());
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
    const func_op = try m.Operation.resource.fetch(env, args[0]);
    const sp_ops = try decodeSafepointOps(env, args[1]);
    defer beam.allocator.free(sp_ops);
    const exclude_entry_args = try beam.get_bool(env, args[2]);
    var collector = SafepointCollector.init(env);
    defer collector.deinit();
    if (!beaverAnalyzeSafepoints(null, func_op, sp_ops.ptr, sp_ops.len, exclude_entry_args, SafepointCollector.collect, &collector)) {
        return error.InvalidSafepointAnalysis;
    }
    return collector.finish();
}

pub fn safepoint_lower_resumable(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const func_op = try m.Operation.resource.fetch(env, args[0]);
    const sp_ops = try decodeSafepointOps(env, args[1]);
    defer beam.allocator.free(sp_ops);
    const save_op = try string_ref.get_binary_as_string_ref(env, args[2]);
    const load_op = try string_ref.get_binary_as_string_ref(env, args[3]);
    var state_arg_index: isize = -1;
    var frame_arg_index: isize = -1;
    var slot_stride: isize = 0;
    var tuple_mode = false;
    var tup_arity: c_int = 0;
    var tup_terms: [*c]const beam.term = undefined;
    if (e.enif_get_tuple(env, args[4], &tup_arity, &tup_terms) != 0 and tup_arity == 3) {
        tuple_mode = true;
        state_arg_index = try beam.get(isize, env, tup_terms[0]);
        frame_arg_index = try beam.get(isize, env, tup_terms[1]);
        slot_stride = try beam.get(isize, env, tup_terms[2]);
    } else {
        state_arg_index = try beam.get(isize, env, args[4]);
    }
    var frame_bytes: u64 = 0;
    if (!beaverLowerResumableCFG(func_op, sp_ops.ptr, sp_ops.len, save_op, load_op, state_arg_index, frame_arg_index, slot_stride, &frame_bytes)) {
        return error.ResumableCFGLoweringFailed;
    }
    if (tuple_mode) {
        return beam.make_ok_term(env, beam.make_u64(env, frame_bytes));
    }
    return beam.make_ok(env);
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
