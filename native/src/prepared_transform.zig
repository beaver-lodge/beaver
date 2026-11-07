const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const prelude = @import("prelude.zig");
const c = prelude.c;
const m = @import("mlir_capi.zig");
const Prepared = struct {
    module: m.Module.T,
    sequence: m.Operation.T,
    context: m.Context.T,
    owner: beam.pid,
    mutex: @import("mutex.zig").Mutex = .{},
    fn close(self: *@This()) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.module.ptr != null) c.mlirModuleDestroy(self.module);
        self.module.ptr = null;
    }
    fn destroy(self: *@This()) void {
        self.close();
        self.mutex.destroy();
    }
    fn checkOwner(self: *@This(), pid: beam.pid) !void {
        if (e.enif_compare_pids(&self.owner, &pid) != 0) return error.TransformOwnerMismatch;
    }
};
const Resource = kinda.RawResourceType(Prepared, "Beaver.MLIR.Transform.Prepared", kinda.resourceDestructor(Prepared, Prepared.destroy));
pub fn transform_prepare(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const module = try m.Module.resource.fetch(env, args[0]);
    const sequence = try m.Operation.resource.fetch(env, args[1]);
    const owner = try beam.self(env);
    const prepared = try Resource.alloc();
    prepared.* = .{ .module = module, .sequence = sequence, .context = c.mlirModuleGetContext(module), .owner = owner };
    defer Resource.release(prepared);
    return e.enif_make_resource(env, prepared);
}
pub fn transform_close(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const prepared = try Resource.fetch(env, args[0]);
    try prepared.checkOwner(try beam.self(env));
    prepared.close();
    return beam.make_ok(env);
}
pub fn apply(env: beam.env, ctx: m.Context.T, arguments: beam.term, caller: beam.pid) !beam.term {
    const args = try @import("execution.zig").tuple(env, arguments, 4);
    const prepared = try Resource.fetch(env, args[0]);
    try prepared.checkOwner(caller);
    const payload = try m.Operation.resource.fetch(env, args[1]);
    if (!c.mlirContextEqual(ctx, prepared.context) or !c.mlirContextEqual(ctx, c.mlirOperationGetContext(payload))) return error.ContextMismatch;
    const checks = try beam.get_bool(env, args[2]);
    const single = try beam.get_bool(env, args[3]);
    prepared.mutex.lock();
    defer prepared.mutex.unlock();
    if (prepared.module.ptr == null) return error.ClosedPreparedTransform;
    const options = c.mlirTransformOptionsCreate();
    defer c.mlirTransformOptionsDestroy(options);
    c.mlirTransformOptionsEnableExpensiveChecks(options, checks);
    c.mlirTransformOptionsEnforceSingleTopLevelTransformOp(options, single);
    const f = kinda.BangFunc(prelude.allKinds, c, "mlirTransformApplyNamedSequence").wrap_ret_call;
    return @import("diagnostic.zig").call_with_diagnostics(env, ctx, f, .{ env, .{ payload, prepared.sequence, c.mlirModuleGetOperation(prepared.module), options } });
}
pub fn open(env: beam.env) void {
    Resource.open(env);
}
pub const nifs = .{
    prelude.beaverRawNIF(@This(), "transform_prepare", 2),
    prelude.beaverRawNIFDirtyCPU(@This(), "transform_close", 1),
};
