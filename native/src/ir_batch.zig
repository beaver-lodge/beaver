const std = @import("std");
const protocol = @import("batch_protocol.zig");
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

const OperandRef = struct {
    kind: protocol.OperandKind,
    index_a: usize = 0,
    index_b: usize = 0,
    value: m.Value.T = .{ .ptr = null },
};

const BuildSpec = struct {
    block_idx: usize,
    name: m.StringRef.T,
    loc: m.Location.T,
    operands: []OperandRef,
    result_types: []m.Type.T,
    attributes: []c.MlirNamedAttribute,
    successors: []m.Block.T,
    num_regions: usize,
};

fn decodeType(env: beam.env, ctx: m.Context.T, term: beam.term) !m.Type.T {
    if (string_ref.get_binary_as_string_ref(env, term)) |sr| {
        if (sr.length == 0) return error.InvalidBatchType;
        const parsed = c.mlirTypeParseGet(ctx, sr);
        if (parsed.ptr == null) return error.InvalidBatchType;
        return parsed;
    } else |_| {
        const t = try m.Type.resource.fetch(env, term);
        if (t.ptr == null or !c.mlirContextEqual(ctx, c.mlirTypeGetContext(t))) return error.ContextMismatch;
        return t;
    }
}

const AttributeTag = enum { unit, i8, i16, i32, i64, index, bool, i1, f32, f64, string, symbol, flat_symbol, type, dense_i32, dense_i64, parse, int };

fn attributeTag(env: beam.env, term: beam.term) !AttributeTag {
    inline for (@typeInfo(AttributeTag).@"enum".fields) |field| {
        if (e.enif_is_identical(term, beam.make_atom(env, field.name)) != 0) return @field(AttributeTag, field.name);
    }
    return error.InvalidBatchAttribute;
}

fn decodeAttribute(env: beam.env, ctx: m.Context.T, term: beam.term) !m.Attribute.T {
    inline for (.{ "unit", "true", "false" }, 0..) |atom, i| {
        if (e.enif_is_identical(term, beam.make_atom(env, atom)) != 0)
            return if (i == 0) c.mlirUnitAttrGet(ctx) else c.mlirBoolAttrGet(ctx, if (i == 1) 1 else 0);
    }
    var arity: c_int = 0;
    var fields: [*c]const beam.term = undefined;
    if (e.enif_get_tuple(env, term, &arity, &fields) != 0 and arity >= 1) {
        const tag = try attributeTag(env, fields[0]);
        const expected: c_int = switch (tag) {
            .unit => 1,
            .int => 3,
            else => 2,
        };
        if (arity != expected) return error.InvalidBatchAttribute;
        return switch (tag) {
            .unit => c.mlirUnitAttrGet(ctx),
            inline .i8, .i16, .i32, .i64 => |integer| c.mlirIntegerAttrGet(
                c.mlirIntegerTypeGet(ctx, comptime std.fmt.parseInt(c_uint, @tagName(integer)[1..], 10) catch unreachable),
                try beam.get(i64, env, fields[1]),
            ),
            .index => c.mlirIntegerAttrGet(c.mlirIndexTypeGet(ctx), try beam.get(i64, env, fields[1])),
            .int => blk: {
                const width = try beam.get(c_uint, env, fields[1]);
                if (width == 0) return error.InvalidBatchAttribute;
                break :blk c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, width), try beam.get(i64, env, fields[2]));
            },
            .bool, .i1 => blk: {
                const value: i64 = if (beam.get_bool(env, fields[1])) |b| @intFromBool(b) else |_| try beam.get(i64, env, fields[1]);
                if (value != 0 and value != 1) return error.InvalidBatchAttribute;
                break :blk c.mlirBoolAttrGet(ctx, @intCast(value));
            },
            inline .f32, .f64 => |float| c.mlirFloatAttrDoubleGet(ctx, @field(c, "mlirF" ++ @tagName(float)[1..] ++ "TypeGet")(ctx), try beam.get(f64, env, fields[1])),
            .string => c.mlirStringAttrGet(ctx, try string_ref.get_binary_as_string_ref(env, fields[1])),
            .symbol, .flat_symbol => blk: {
                const symbol = try string_ref.get_binary_as_string_ref(env, fields[1]);
                if (symbol.length == 0) return error.InvalidBatchAttribute;
                break :blk c.mlirFlatSymbolRefAttrGet(ctx, symbol);
            },
            .type => c.mlirTypeAttrGet(try decodeType(env, ctx, fields[1])),
            inline .dense_i32, .dense_i64 => |dense| blk: {
                const width = @tagName(dense)[7..];
                const Int = std.meta.Int(.signed, comptime std.fmt.parseInt(u16, width, 10) catch unreachable);
                const elems = try beam.allocator.alloc(Int, try beam.get_list_length(env, fields[1]));
                defer beam.allocator.free(elems);
                var rest = fields[1];
                for (elems) |*el| el.* = try beam.get(Int, env, try beam.get_head_and_iter(env, &rest));
                break :blk @field(c, "mlirDenseI" ++ width ++ "ArrayGet")(ctx, @intCast(elems.len), elems.ptr);
            },
            .parse => parseAttribute(ctx, try string_ref.get_binary_as_string_ref(env, fields[1])),
        };
    }
    if (string_ref.get_binary_as_string_ref(env, term)) |source| return parseAttribute(ctx, source) else |_| {
        const attr = try m.Attribute.resource.fetch(env, term);
        if (attr.ptr == null or !c.mlirContextEqual(ctx, c.mlirAttributeGetContext(attr))) return error.ContextMismatch;
        return attr;
    }
}

fn parseAttribute(ctx: m.Context.T, source: m.StringRef.T) !m.Attribute.T {
    if (source.length == 0) return error.InvalidBatchAttribute;
    const parsed = c.mlirAttributeParseGet(ctx, source);
    if (parsed.ptr == null) return error.InvalidBatchAttribute;
    return parsed;
}

pub fn build(env: beam.env, ctx: m.Context.T, arguments: beam.term) !beam.term {
    var memory = std.heap.ArenaAllocator.init(beam.allocator);
    defer memory.deinit();
    const allocator = memory.allocator();
    const args = try tuple(env, arguments, 4);

    var insert_mode: u8 = 0; // 0 = end, 1 = start, 2 = before, 3 = after, 4 = multi-block end
    var anchor_op: m.Operation.T = .{ .ptr = null };
    var blocks: []m.Block.T = undefined;
    if (e.enif_is_identical(args[1], beam.make_atom(env, "blocks")) != 0) {
        insert_mode = 4;
        blocks = try @import("callback_args.zig").Range(m.Block).decode(env, args[0], allocator);
        if (blocks.len == 0) return error.InvalidBatchBlock;
        for (blocks) |block| if (block.ptr == null) return error.InvalidBatchBlock;
    } else {
        const single_block = try m.Block.resource.fetch(env, args[0]);
        if (single_block.ptr == null) return error.InvalidBatchBlock;
        blocks = try allocator.alloc(m.Block.T, 1);
        blocks[0] = single_block;

        if (e.enif_is_identical(args[1], beam.make_atom(env, "end")) != 0) {
            insert_mode = 0;
        } else if (e.enif_is_identical(args[1], beam.make_atom(env, "start")) != 0) {
            insert_mode = 1;
        } else {
            const pos_tuple = try tuple(env, args[1], 2);
            anchor_op = try operation(env, ctx, pos_tuple[1]);
            if (!c.mlirBlockEqual(c.mlirOperationGetBlock(anchor_op), single_block)) return error.OperationOutsideTargetBlock;
            if (e.enif_is_identical(pos_tuple[0], beam.make_atom(env, "before")) != 0) {
                insert_mode = 2;
            } else if (e.enif_is_identical(pos_tuple[0], beam.make_atom(env, "after")) != 0) {
                insert_mode = 3;
            } else return error.InvalidInsertionPosition;
        }
    }

    const count = try beam.get_list_length(env, args[2]);
    if (count > 1048576) return error.BatchTooLarge;

    const verify_root = if (beam.is_nil2(env, args[3])) m.Operation.T{ .ptr = null } else try operation(env, ctx, args[3]);
    const binding = if (verify_root.ptr != null) c.beaverExecutionBindOperation(verify_root) else null;
    defer c.beaverExecutionUnbindOperation(binding);

    const specs = try allocator.alloc(BuildSpec, count);
    var rest = args[2];
    for (specs, 0..) |*spec, i| {
        const spec_term = try beam.get_head_and_iter(env, &rest);
        const fields = try protocol.Operation.decode(env, spec_term);
        const block_idx = if (fields.optional("block")) |value| try beam.get(usize, env, value) else 0;
        if (block_idx >= blocks.len) return error.InvalidBatchBlock;
        const num_block_args: usize = @intCast(c.mlirBlockGetNumArguments(blocks[block_idx]));

        const name = try string_ref.get_binary_as_string_ref(env, fields.get("name"));
        if (name.length == 0) return error.EmptyOperationName;
        const loc = try m.Location.resource.fetch(env, fields.get("location"));
        if (loc.ptr == null or !c.mlirContextEqual(ctx, c.mlirLocationGetContext(loc))) return error.ContextMismatch;

        const op_len = try beam.get_list_length(env, fields.get("operands"));
        const op_refs = try allocator.alloc(OperandRef, op_len);

        var op_rest = fields.get("operands");
        for (op_refs) |*op_ref| {
            const o_fields = try protocol.Operand.decode(env, try beam.get_head_and_iter(env, &op_rest));
            const kind = try protocol.operandKind(env, o_fields.get("kind"));
            switch (kind) {
                .value => {
                    const val = try m.Value.resource.fetch(env, o_fields.get("value"));
                    if (val.ptr == null or !c.mlirContextEqual(ctx, c.mlirTypeGetContext(c.mlirValueGetType(val)))) return error.ContextMismatch;
                    op_ref.* = .{ .kind = .value, .value = val };
                },
                .result => {
                    const op_idx = try beam.get(usize, env, o_fields.get("index"));
                    const res_idx = try beam.get(usize, env, o_fields.get("value"));
                    if (op_idx >= i or res_idx >= specs[op_idx].result_types.len) return error.InvalidBatchResultReference;
                    op_ref.* = .{ .kind = .result, .index_a = op_idx, .index_b = res_idx };
                },
                .block_arg => {
                    const arg_idx = try beam.get(usize, env, o_fields.get("index"));
                    if (arg_idx >= num_block_args) return error.InvalidBlockArgumentIndex;
                    op_ref.* = .{ .kind = .block_arg, .index_a = arg_idx };
                },
                .cfg_block_arg => {
                    const target_blk = try beam.get(usize, env, o_fields.get("index"));
                    const arg_idx = try beam.get(usize, env, o_fields.get("value"));
                    if (target_blk >= blocks.len or arg_idx >= @as(usize, @intCast(c.mlirBlockGetNumArguments(blocks[target_blk])))) {
                        return error.InvalidBlockArgumentIndex;
                    }
                    op_ref.* = .{ .kind = .cfg_block_arg, .index_a = target_blk, .index_b = arg_idx };
                },
            }
        }

        const res_len = try beam.get_list_length(env, fields.get("result_types"));
        const res_types = try allocator.alloc(m.Type.T, res_len);

        var res_rest = fields.get("result_types");
        for (res_types) |*rt| {
            rt.* = try decodeType(env, ctx, try beam.get_head_and_iter(env, &res_rest));
        }

        const attr_len = try beam.get_list_length(env, fields.get("attributes"));
        const attrs = try allocator.alloc(c.MlirNamedAttribute, attr_len);

        var attr_rest = fields.get("attributes");
        for (attrs) |*na| {
            const a_fields = try tuple(env, try beam.get_head_and_iter(env, &attr_rest), 2);
            const a_name = try string_ref.get_binary_as_string_ref(env, a_fields[0]);
            const a_val = try decodeAttribute(env, ctx, a_fields[1]);
            na.* = c.mlirNamedAttributeGet(c.mlirIdentifierGet(ctx, a_name), a_val);
        }

        const succs = try @import("callback_args.zig").Range(m.Block).decode(env, fields.get("successors"), allocator);
        for (succs) |block| if (block.ptr == null) return error.InvalidBatchSuccessor;

        const num_regions = try beam.get(usize, env, fields.get("regions"));
        spec.* = .{
            .block_idx = block_idx,
            .name = name,
            .loc = loc,
            .operands = op_refs,
            .result_types = res_types,
            .attributes = attrs,
            .successors = succs,
            .num_regions = num_regions,
        };
    }

    const created_ops = try allocator.alloc(m.Operation.T, count);

    const op_terms = try allocator.alloc(beam.term, count);

    const res_list_terms = try allocator.alloc(beam.term, count);

    var prev_inserted: m.Operation.T = anchor_op;
    for (specs, 0..) |spec, i| {
        const spec_block = blocks[spec.block_idx];
        const resolved_operands = try beam.allocator.alloc(m.Value.T, spec.operands.len);
        defer beam.allocator.free(resolved_operands);
        for (spec.operands, 0..) |op_ref, k| {
            resolved_operands[k] = switch (op_ref.kind) {
                .value => op_ref.value,
                .result => c.mlirOperationGetResult(created_ops[op_ref.index_a], @intCast(op_ref.index_b)),
                .block_arg => c.mlirBlockGetArgument(spec_block, @intCast(op_ref.index_a)),
                .cfg_block_arg => c.mlirBlockGetArgument(blocks[op_ref.index_a], @intCast(op_ref.index_b)),
            };
        }

        var state = c.mlirOperationStateGet(spec.name, spec.loc);
        inline for (.{ "Operands", "Results", "Attributes", "Successors" }, .{ resolved_operands, spec.result_types, spec.attributes, spec.successors }) |kind, items| {
            if (items.len > 0) @field(c, "mlirOperationStateAdd" ++ kind)(&state, @intCast(items.len), items.ptr);
        }
        if (spec.num_regions > 0) {
            const regions = try beam.allocator.alloc(m.Region.T, spec.num_regions);
            defer beam.allocator.free(regions);
            for (regions) |*r| r.* = c.mlirRegionCreate();
            c.mlirOperationStateAddOwnedRegions(&state, @intCast(regions.len), regions.ptr);
        }

        const created = c.beaverOperationCreate(state);
        if (created.ptr == null) return error.OperationCreationFailed;
        switch (insert_mode) {
            0, 4 => c.mlirBlockAppendOwnedOperation(spec_block, created),
            1 => {
                if (i == 0) {
                    c.mlirBlockInsertOwnedOperation(spec_block, 0, created);
                } else {
                    c.mlirBlockInsertOwnedOperationAfter(spec_block, prev_inserted, created);
                }
                prev_inserted = created;
            },
            2 => c.mlirBlockInsertOwnedOperationBefore(spec_block, anchor_op, created),
            3 => {
                c.mlirBlockInsertOwnedOperationAfter(spec_block, prev_inserted, created);
                prev_inserted = created;
            },
            else => unreachable,
        }

        created_ops[i] = created;
        op_terms[i] = try m.Operation.resource.make_kind(env, created);
        res_list_terms[i] = try values(env, created, false, false);
    }

    const valid = verify_root.ptr == null or c.mlirOperationVerify(verify_root);
    var out = [_]beam.term{
        beam.make_bool(env, valid),
        beam.make_term_list(env, op_terms),
        beam.make_term_list(env, res_list_terms),
    };
    return beam.make_tuple(env, &out);
}

const SymListCtx = struct {
    env: beam.env,
    list: std.ArrayListUnmanaged(beam.term) = .empty,
    oom: bool = false,
};

fn appendSymCallback(sr: m.StringRef.T, user_data: ?*anyopaque) callconv(.c) void {
    const ctx_ptr: *SymListCtx = @ptrCast(@alignCast(user_data.?));
    if (ctx_ptr.oom) return;
    const slice = if (sr.length == 0) "" else sr.data[0..@intCast(sr.length)];
    ctx_ptr.list.append(beam.allocator, beam.make_slice(ctx_ptr.env, slice)) catch {
        ctx_ptr.oom = true;
    };
}

pub fn referencedSymbols(env: beam.env, ctx: m.Context.T, arguments: beam.term) !beam.term {
    const args = try tuple(env, arguments, 1);
    const root = try operation(env, ctx, args[0]);
    var cb_ctx = SymListCtx{ .env = env };
    defer cb_ctx.list.deinit(beam.allocator);
    c.beaverSymbolTableWalkReferencedSymbols(root, appendSymCallback, &cb_ctx);
    if (cb_ctx.oom) return error.OutOfMemory;
    return beam.make_term_list(env, cb_ctx.list.items);
}
