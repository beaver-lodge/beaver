const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const c = @import("prelude.zig").c;
const m = @import("mlir_capi.zig");
const tuple = @import("execution.zig").tuple;
const string_ref = @import("string_ref.zig");

const Constraint = extern struct {
    kind: c_int,
    index: isize,
    other: isize,
    name: m.StringRef.T = .{ .data = null, .length = 0 },
    attribute: m.Attribute.T = .{ .ptr = null },
    type: m.Type.T = .{ .ptr = null },
};
const Edit = extern struct {
    kind: c_int,
    index: isize,
    name: m.StringRef.T = .{ .data = null, .length = 0 },
    attribute: m.Attribute.T = .{ .ptr = null },
};
extern fn beaverDeclarativePatternAdd(m.RewritePatternSet.T, m.Context.T, m.StringRef.T, c_uint, [*]const Constraint, usize, [*]const Edit, usize) void;

fn attribute(env: beam.env, ctx: m.Context.T, term: beam.term) !m.Attribute.T {
    const attr = try m.Attribute.resource.fetch(env, term);
    if (attr.ptr == null or !c.mlirContextEqual(ctx, c.mlirAttributeGetContext(attr))) return error.ContextMismatch;
    return attr;
}

pub fn declarative_pattern_add(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const set = try m.RewritePatternSet.resource.fetch(env, args[0]);
    const ctx = try m.Context.resource.fetch(env, args[1]);
    const root = try string_ref.get_binary_as_string_ref(env, args[2]);
    if (root.length == 0) return error.EmptyPatternRoot;
    const benefit = try beam.get(c_uint, env, args[3]);
    if (benefit >= 65535) return error.InvalidPatternBenefit;
    const checks = try beam.allocator.alloc(Constraint, try beam.get_list_length(env, args[4]));
    defer beam.allocator.free(checks);
    var rest = args[4];
    for (checks) |*check| {
        const fields = try tuple(env, try beam.get_head_and_iter(env, &rest), 4);
        check.* = .{ .kind = try beam.get(c_int, env, fields[0]), .index = try beam.get(isize, env, fields[1]), .other = 0 };
        if (check.index < 0) return error.InvalidPatternIndex;
        switch (check.kind) {
            0, 1 => {},
            2 => {
                check.name = try string_ref.get_binary_as_string_ref(env, fields[2]);
                check.attribute = try attribute(env, ctx, fields[3]);
            },
            3, 4 => {
                check.type = try m.Type.resource.fetch(env, fields[3]);
                if (check.type.ptr == null or !c.mlirContextEqual(ctx, c.mlirTypeGetContext(check.type))) return error.ContextMismatch;
            },
            5 => {
                check.other = try beam.get(isize, env, fields[3]);
                if (check.other < 0) return error.InvalidPatternIndex;
            },
            else => return error.InvalidPatternConstraint,
        }
    }
    const edits = try beam.allocator.alloc(Edit, try beam.get_list_length(env, args[5]));
    defer beam.allocator.free(edits);
    if (edits.len == 0) return error.EmptyPatternRewrite;
    rest = args[5];
    for (edits, 0..) |*edit, index| {
        const fields = try tuple(env, try beam.get_head_and_iter(env, &rest), 4);
        edit.* = .{ .kind = try beam.get(c_int, env, fields[0]), .index = try beam.get(isize, env, fields[1]) };
        switch (edit.kind) {
            0 => {
                edit.name = try string_ref.get_binary_as_string_ref(env, fields[2]);
                edit.attribute = try attribute(env, ctx, fields[3]);
                for (edits[0..index]) |prior| {
                    if (prior.kind == 0 and c.mlirStringRefEqual(prior.name, edit.name)) return error.DuplicatePatternAttribute;
                }
            },
            1 => {
                if (edit.index < 0 or edits.len != 1) return error.InvalidPatternRewrite;
            },
            2 => {
                if (edits.len != 1) return error.InvalidPatternRewrite;
            },
            else => return error.InvalidPatternRewrite,
        }
    }
    beaverDeclarativePatternAdd(set, ctx, root, benefit, checks.ptr, checks.len, edits.ptr, edits.len);
    return beam.make_ok(env);
}

pub const nifs = .{@import("prelude.zig").beaverRawNIF(@This(), "declarative_pattern_add", 6)};
