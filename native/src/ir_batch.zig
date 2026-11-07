const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
const c = @import("prelude.zig").c;
const m = @import("mlir_capi.zig");
const string_ref = @import("string_ref.zig");
const tuple = @import("execution.zig").tuple;

fn operation(env: beam.env, ctx: m.Context.T, term: beam.term) !m.Operation.T {
    const op = try m.Operation.resource.fetch(env, term);
    if (op.ptr == null or !c.mlirContextEqual(ctx, c.mlirOperationGetContext(op))) return error.ContextMismatch;
    return op;
}

fn values(env: beam.env, op: m.Operation.T, comptime operands: bool, comptime types: bool) !beam.term {
    const count: usize = @intCast(if (operands) c.mlirOperationGetNumOperands(op) else c.mlirOperationGetNumResults(op));
    const terms = try beam.allocator.alloc(beam.term, count);
    defer beam.allocator.free(terms);
    for (terms, 0..) |*term, i| {
        const value = if (operands) c.mlirOperationGetOperand(op, @intCast(i)) else c.mlirOperationGetResult(op, @intCast(i));
        term.* = if (types) try m.Type.resource.make_kind(env, c.mlirValueGetType(value)) else try m.Value.resource.make_kind(env, value);
    }
    return beam.make_term_list(env, terms);
}

pub fn snapshot(env: beam.env, ctx: m.Context.T, arguments: beam.term) !beam.term {
    const args = try tuple(env, arguments, 2);
    const count = try beam.get_list_length(env, args[0]);
    if (count > 256) return error.BatchTooLarge;
    const attr_count = try beam.get_list_length(env, args[1]);
    const records = try beam.allocator.alloc(beam.term, count);
    defer beam.allocator.free(records);
    const attrs = try beam.allocator.alloc(beam.term, attr_count);
    defer beam.allocator.free(attrs);
    var rest = args[0];
    for (records) |*record| {
        const raw = try beam.get_head_and_iter(env, &rest);
        const op = try operation(env, ctx, raw);
        var names = args[1];
        for (attrs) |*attr| {
            const name = try string_ref.get_binary_as_string_ref(env, try beam.get_head_and_iter(env, &names));
            const value = c.mlirOperationGetAttributeByName(op, name);
            attr.* = if (value.ptr == null) beam.make_nil(env) else try m.Attribute.resource.make_kind(env, value);
        }
        const name = c.mlirIdentifierStr(c.mlirOperationGetName(op));
        var fields = [_]beam.term{
            try m.Operation.resource.make_kind(env, op), beam.make_slice(env, name.data[0..@intCast(name.length)]),
            try values(env, op, true, false),            try values(env, op, false, false),
            try values(env, op, true, true),             try values(env, op, false, true),
            beam.make_term_list(env, attrs),
        };
        record.* = beam.make_tuple(env, &fields);
    }
    return beam.make_term_list(env, records);
}

const Edit = struct {
    op: m.Operation.T,
    name: m.StringRef.T = .{ .data = null, .length = 0 },
    attribute: m.Attribute.T = .{ .ptr = null },
    operand: m.Value.T = .{ .ptr = null },
    index: isize = -1,
};

pub fn update(env: beam.env, ctx: m.Context.T, arguments: beam.term) !beam.term {
    const args = try tuple(env, arguments, 3);
    const root = try operation(env, ctx, args[0]);
    const binding = c.beaverExecutionBindOperation(root);
    defer c.beaverExecutionUnbindOperation(binding);
    const count = try beam.get_list_length(env, args[1]);
    if (count > 256) return error.BatchTooLarge;
    const edits = try beam.allocator.alloc(Edit, count);
    defer beam.allocator.free(edits);
    var rest = args[1];
    // Validate the entire batch before the first mutation. These edits do not
    // erase or replace operations, so later handles cannot be invalidated.
    for (edits) |*edit| {
        const fields = try tuple(env, try beam.get_head_and_iter(env, &rest), 4);
        edit.* = .{ .op = try operation(env, ctx, fields[1]) };
        var ancestor = edit.op;
        while (ancestor.ptr != null and !c.mlirOperationEqual(ancestor, root)) ancestor = c.mlirOperationGetParentOperation(ancestor);
        if (ancestor.ptr == null) return error.OperationOutsideBatchRoot;
        if (e.enif_is_identical(fields[0], beam.make_atom(env, "operand")) != 0) {
            edit.index = try beam.get(isize, env, fields[2]);
            if (edit.index < 0 or edit.index >= c.mlirOperationGetNumOperands(edit.op)) return error.InvalidOperandIndex;
            edit.operand = try m.Value.resource.fetch(env, fields[3]);
            if (edit.operand.ptr == null or !c.mlirTypeEqual(c.mlirValueGetType(edit.operand), c.mlirValueGetType(c.mlirOperationGetOperand(edit.op, edit.index)))) return error.OperandTypeMismatch;
        } else if (e.enif_is_identical(fields[0], beam.make_atom(env, "attribute")) != 0) {
            edit.name = try string_ref.get_binary_as_string_ref(env, fields[2]);
            edit.attribute = try m.Attribute.resource.fetch(env, fields[3]);
            if (edit.attribute.ptr == null or !c.mlirContextEqual(ctx, c.mlirAttributeGetContext(edit.attribute))) return error.ContextMismatch;
            const previous = c.mlirOperationGetAttributeByName(edit.op, edit.name);
            if (c.beaverOperationHasInherentAttributeByName(edit.op, edit.name) and
                (previous.ptr == null or !c.mlirTypeIDEqual(c.mlirAttributeGetTypeID(previous), c.mlirAttributeGetTypeID(edit.attribute)))) return error.InherentAttributeTypeMismatch;
        } else return error.UnsupportedBatchEdit;
    }
    const rewriter = c.mlirIRRewriterCreate(ctx);
    defer c.mlirIRRewriterDestroy(rewriter);
    const base = rewriter;
    for (edits) |edit| {
        c.mlirRewriterBaseStartOpModification(base, edit.op);
        if (edit.index >= 0) c.mlirOperationSetOperand(edit.op, edit.index, edit.operand) else c.mlirOperationSetAttributeByName(edit.op, edit.name, edit.attribute);
        c.mlirRewriterBaseFinalizeOpModification(base, edit.op);
    }
    const verify = e.enif_is_identical(args[2], beam.make_atom(env, "true")) != 0;
    return beam.make_bool(env, !verify or c.mlirOperationVerify(root));
}
