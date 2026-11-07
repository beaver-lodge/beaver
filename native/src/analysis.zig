const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const m = @import("mlir_capi.zig");
const tuple = @import("execution.zig").tuple;

const Session = struct {
    handle: ?*anyopaque,
    context: m.Context.T,
    owner: beam.pid,
    mutation_depth: u32 = 0,
    mutex: @import("mutex.zig").Mutex = .{},

    fn close(self: *@This()) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.releaseHandle();
    }
    fn releaseHandle(self: *@This()) void {
        if (self.handle) |handle| c.beaverAnalysisDestroy(handle);
        self.handle = null;
    }
    fn destroy(self: *@This()) void {
        self.close();
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
    const session = try Resource.alloc();
    session.* = .{ .handle = null, .context = c.mlirOperationGetContext(root), .owner = owner };
    defer Resource.release(session);
    session.handle = c.beaverAnalysisCreate(root);
    return e.enif_make_resource(env, session);
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
pub fn open(env: beam.env) void {
    Resource.open(env);
}
pub const nifs = .{
    prelude.beaverRawNIF(@This(), "analysis_create", 1),
    prelude.beaverRawNIFDirtyCPU(@This(), "analysis_close", 1),
    prelude.beaverRawNIFDirtyCPU(@This(), "analysis_invalidate", 2),
    prelude.beaverRawNIFDirtyCPU(@This(), "analysis_mutation", 2),
};
