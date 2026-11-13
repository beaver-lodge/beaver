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
    extra: m.StringRef.T = .{ .data = null, .length = 0 },
    attribute: m.Attribute.T = .{ .ptr = null },
    type: m.Type.T = .{ .ptr = null },
};
const Edit = extern struct {
    kind: c_int,
    index: isize,
    other: isize = 0,
    name: m.StringRef.T = .{ .data = null, .length = 0 },
    extra: m.StringRef.T = .{ .data = null, .length = 0 },
    attribute: m.Attribute.T = .{ .ptr = null },
    type: m.Type.T = .{ .ptr = null },
};
extern fn beaverDeclarativePatternAdd(m.RewritePatternSet.T, m.Context.T, m.TypeConverter.T, m.StringRef.T, c_uint, [*]const Constraint, usize, [*]const Edit, usize) bool;
extern fn beaverRewritePatternSetAddPDL(m.RewritePatternSet.T, m.Operation.T) bool;
fn attribute(env: beam.env, ctx: m.Context.T, term: beam.term) !m.Attribute.T {
    const attr = try m.Attribute.resource.fetch(env, term);
    if (attr.ptr == null or !c.mlirContextEqual(ctx, c.mlirAttributeGetContext(attr))) return error.ContextMismatch;
    return attr;
}
fn fetchType(env: beam.env, ctx: m.Context.T, term: beam.term) !m.Type.T {
    const t = try m.Type.resource.fetch(env, term);
    if (t.ptr == null or !c.mlirContextEqual(ctx, c.mlirTypeGetContext(t))) return error.ContextMismatch;
    return t;
}
pub fn declarative_pattern_add(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const set = try m.RewritePatternSet.resource.fetch(env, args[0]);
    var ctx: m.Context.T = .{ .ptr = null };
    var converter: m.TypeConverter.T = .{ .ptr = null };
    var tup_arity: c_int = 0;
    var tup_terms: [*c]const beam.term = undefined;
    if (e.enif_get_tuple(env, args[1], &tup_arity, &tup_terms) != 0 and tup_arity == 2) {
        ctx = try m.Context.resource.fetch(env, tup_terms[0]);
        if (!beam.is_nil2(env, tup_terms[1])) {
            converter = try m.TypeConverter.resource.fetch(env, tup_terms[1]);
        }
    } else {
        ctx = try m.Context.resource.fetch(env, args[1]);
    }
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
                if (string_ref.get_binary_as_string_ref(env, fields[3])) |s| {
                    check.extra = s;
                } else |_| {
                    check.attribute = try attribute(env, ctx, fields[3]);
                }
            },
            3, 4 => {
                if (string_ref.get_binary_as_string_ref(env, fields[3])) |s| {
                    check.extra = s;
                } else |_| {
                    check.type = try fetchType(env, ctx, fields[3]);
                }
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
            0, 5 => {
                edit.name = try string_ref.get_binary_as_string_ref(env, fields[2]);
                if (string_ref.get_binary_as_string_ref(env, fields[3])) |s| {
                    edit.extra = s;
                } else |_| {
                    edit.attribute = try attribute(env, ctx, fields[3]);
                }
                for (edits[0..index]) |prior| {
                    if (prior.kind == edit.kind and c.mlirStringRefEqual(prior.name, edit.name)) return error.DuplicatePatternAttribute;
                }
            },
            1 => {
                if (edit.index < 0 or edits.len != 1) return error.InvalidPatternRewrite;
            },
            2 => {
                if (edits.len != 1) return error.InvalidPatternRewrite;
            },
            3 => {
                if (index != 0) return error.InvalidPatternRewrite;
                edit.name = try string_ref.get_binary_as_string_ref(env, fields[2]);
                if (edit.name.length == 0) return error.InvalidPatternRewrite;
                edit.other = try beam.get(isize, env, fields[3]);
            },
            4 => {
                if (index == 0 or edit.index < 0) return error.InvalidPatternRewrite;
            },
            6, 13 => {
                if (index == 0) return error.InvalidPatternRewrite;
                edit.name = try string_ref.get_binary_as_string_ref(env, fields[2]);
            },
            14 => {
                if (index == 0) return error.InvalidPatternRewrite;
                edit.name = try string_ref.get_binary_as_string_ref(env, fields[2]);
                edit.extra = try string_ref.get_binary_as_string_ref(env, fields[3]);
                if (edit.name.length == 0 or edit.extra.length == 0) return error.InvalidPatternRewrite;
            },
            7, 12 => {
                if (index == 0) return error.InvalidPatternRewrite;
                if (edit.index != -1) {
                    if (string_ref.get_binary_as_string_ref(env, fields[3])) |s| {
                        edit.name = s;
                    } else |_| {
                        edit.type = try fetchType(env, ctx, fields[3]);
                    }
                }
            },
            8 => {
                if (index != 0) return error.InvalidPatternRewrite;
                edit.name = try string_ref.get_binary_as_string_ref(env, fields[2]);
                if (edit.name.length == 0) return error.InvalidPatternRewrite;
                const sub = try tuple(env, fields[3], 3);
                edit.other = try beam.get(isize, env, sub[0]);
                if (!beam.is_nil2(env, sub[1])) {
                    edit.extra = try string_ref.get_binary_as_string_ref(env, sub[1]);
                }
                if (!beam.is_nil2(env, sub[2])) {
                    edit.type = try fetchType(env, ctx, sub[2]);
                }
            },
            9 => {
                if (index == 0) return error.InvalidPatternRewrite;
            },
            10, 11 => {
                if (index == 0) return error.InvalidPatternRewrite;
                if (!beam.is_nil2(env, fields[3])) {
                    if (string_ref.get_binary_as_string_ref(env, fields[3])) |s| {
                        edit.name = s;
                    } else |_| {
                        edit.type = try fetchType(env, ctx, fields[3]);
                    }
                }
            },
            else => return error.InvalidPatternRewrite,
        }
    }
    if (!beaverDeclarativePatternAdd(set, ctx, converter, root, benefit, checks.ptr, checks.len, edits.ptr, edits.len)) {
        return error.InvalidPatternProgram;
    }
    return beam.make_ok(env);
}
pub fn pdl_pattern_add(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const set = try m.RewritePatternSet.resource.fetch(env, args[0]);
    const op = try m.Operation.resource.fetch(env, args[1]);
    if (!beaverRewritePatternSetAddPDL(set, op)) return error.InvalidPDLPatternModule;
    return beam.make_ok(env);
}
pub const nifs = .{
    @import("prelude.zig").beaverRawNIF(@This(), "declarative_pattern_add", 6),
    @import("prelude.zig").beaverRawNIF(@This(), "pdl_pattern_add", 2),
};
