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

fn decodeAttribute(env: beam.env, ctx: m.Context.T, term: beam.term) !m.Attribute.T {
    if (e.enif_is_identical(term, beam.make_atom(env, "unit")) != 0) {
        return c.mlirUnitAttrGet(ctx);
    }
    if (e.enif_is_identical(term, beam.make_atom(env, "true")) != 0) {
        return c.mlirBoolAttrGet(ctx, 1);
    }
    if (e.enif_is_identical(term, beam.make_atom(env, "false")) != 0) {
        return c.mlirBoolAttrGet(ctx, 0);
    }

    var tag_arity: c_int = 0;
    var tag_terms: [*c]const beam.term = undefined;
    if (e.enif_get_tuple(env, term, &tag_arity, &tag_terms) != 0 and tag_arity >= 1) {
        const tag = tag_terms[0];
        if (tag_arity == 1 and e.enif_is_identical(tag, beam.make_atom(env, "unit")) != 0) {
            return c.mlirUnitAttrGet(ctx);
        }
        if (tag_arity == 2) {
            if (e.enif_is_identical(tag, beam.make_atom(env, "i64")) != 0) {
                return c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 64), try beam.get(i64, env, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "i32")) != 0) {
                return c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 32), try beam.get(i64, env, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "i16")) != 0) {
                return c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 16), try beam.get(i64, env, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "i8")) != 0) {
                return c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, 8), try beam.get(i64, env, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "index")) != 0) {
                return c.mlirIntegerAttrGet(c.mlirIndexTypeGet(ctx), try beam.get(i64, env, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "bool")) != 0 or e.enif_is_identical(tag, beam.make_atom(env, "i1")) != 0) {
                if (beam.get_bool(env, tag_terms[1])) |b| {
                    return c.mlirBoolAttrGet(ctx, if (b) 1 else 0);
                } else |_| {
                    const n = try beam.get(i64, env, tag_terms[1]);
                    if (n != 0 and n != 1) return error.InvalidBatchAttribute;
                    return c.mlirBoolAttrGet(ctx, @intCast(n));
                }
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "f64")) != 0) {
                return c.mlirFloatAttrDoubleGet(ctx, c.mlirF64TypeGet(ctx), try beam.get(f64, env, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "f32")) != 0) {
                return c.mlirFloatAttrDoubleGet(ctx, c.mlirF32TypeGet(ctx), try beam.get(f64, env, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "string")) != 0) {
                return c.mlirStringAttrGet(ctx, try string_ref.get_binary_as_string_ref(env, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "symbol")) != 0 or e.enif_is_identical(tag, beam.make_atom(env, "flat_symbol")) != 0) {
                const sym = try string_ref.get_binary_as_string_ref(env, tag_terms[1]);
                if (sym.length == 0) return error.InvalidBatchAttribute;
                return c.mlirFlatSymbolRefAttrGet(ctx, sym);
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "type")) != 0) {
                return c.mlirTypeAttrGet(try decodeType(env, ctx, tag_terms[1]));
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "dense_i32")) != 0) {
                const len = try beam.get_list_length(env, tag_terms[1]);
                const elems = try beam.allocator.alloc(i32, len);
                defer beam.allocator.free(elems);
                var list_rest = tag_terms[1];
                for (elems) |*el| el.* = try beam.get(i32, env, try beam.get_head_and_iter(env, &list_rest));
                return c.mlirDenseI32ArrayGet(ctx, @intCast(len), elems.ptr);
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "dense_i64")) != 0) {
                const len = try beam.get_list_length(env, tag_terms[1]);
                const elems = try beam.allocator.alloc(i64, len);
                defer beam.allocator.free(elems);
                var list_rest = tag_terms[1];
                for (elems) |*el| el.* = try beam.get(i64, env, try beam.get_head_and_iter(env, &list_rest));
                return c.mlirDenseI64ArrayGet(ctx, @intCast(len), elems.ptr);
            }
            if (e.enif_is_identical(tag, beam.make_atom(env, "parse")) != 0) {
                const sr = try string_ref.get_binary_as_string_ref(env, tag_terms[1]);
                if (sr.length == 0) return error.InvalidBatchAttribute;
                const parsed = c.mlirAttributeParseGet(ctx, sr);
                if (parsed.ptr == null) return error.InvalidBatchAttribute;
                return parsed;
            }
        }
        if (tag_arity == 3 and e.enif_is_identical(tag, beam.make_atom(env, "int")) != 0) {
            const width = try beam.get(c_uint, env, tag_terms[1]);
            if (width == 0) return error.InvalidBatchAttribute;
            return c.mlirIntegerAttrGet(c.mlirIntegerTypeGet(ctx, width), try beam.get(i64, env, tag_terms[2]));
        }
        return error.InvalidBatchAttribute;
    }

    if (string_ref.get_binary_as_string_ref(env, term)) |sr| {
        if (sr.length == 0) return error.InvalidBatchAttribute;
        const parsed = c.mlirAttributeParseGet(ctx, sr);
        if (parsed.ptr == null) return error.InvalidBatchAttribute;
        return parsed;
    } else |_| {
        const a_val = try m.Attribute.resource.fetch(env, term);
        if (a_val.ptr == null or !c.mlirContextEqual(ctx, c.mlirAttributeGetContext(a_val))) return error.ContextMismatch;
        return a_val;
    }
}

pub fn build(env: beam.env, ctx: m.Context.T, arguments: beam.term) !beam.term {
    const args = try tuple(env, arguments, 4);

    var insert_mode: u8 = 0; // 0 = end, 1 = start, 2 = before, 3 = after, 4 = multi-block end
    var anchor_op: m.Operation.T = .{ .ptr = null };
    var blocks: []m.Block.T = undefined;
    if (e.enif_is_identical(args[1], beam.make_atom(env, "blocks")) != 0) {
        insert_mode = 4;
        const num_blocks = try beam.get_list_length(env, args[0]);
        if (num_blocks == 0) return error.InvalidBatchBlock;
        blocks = try beam.allocator.alloc(m.Block.T, num_blocks);
        errdefer beam.allocator.free(blocks);
        var blk_rest = args[0];
        for (blocks) |*b| {
            b.* = try m.Block.resource.fetch(env, try beam.get_head_and_iter(env, &blk_rest));
            if (b.ptr == null) return error.InvalidBatchBlock;
        }
    } else {
        const single_block = try m.Block.resource.fetch(env, args[0]);
        if (single_block.ptr == null) return error.InvalidBatchBlock;
        blocks = try beam.allocator.alloc(m.Block.T, 1);
        blocks[0] = single_block;
        errdefer beam.allocator.free(blocks);

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
    defer beam.allocator.free(blocks);

    const count = try beam.get_list_length(env, args[2]);
    if (count > 1048576) return error.BatchTooLarge;

    const verify_root = if (beam.is_nil2(env, args[3])) m.Operation.T{ .ptr = null } else try operation(env, ctx, args[3]);
    const binding = if (verify_root.ptr != null) c.beaverExecutionBindOperation(verify_root) else null;
    defer c.beaverExecutionUnbindOperation(binding);

    const specs = try beam.allocator.alloc(BuildSpec, count);
    var initialized_specs: usize = 0;
    defer {
        for (specs[0..initialized_specs]) |spec| {
            beam.allocator.free(spec.operands);
            beam.allocator.free(spec.result_types);
            beam.allocator.free(spec.attributes);
            beam.allocator.free(spec.successors);
        }
        beam.allocator.free(specs);
    }

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
        const op_refs = try beam.allocator.alloc(OperandRef, op_len);
        errdefer beam.allocator.free(op_refs);
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
        const res_types = try beam.allocator.alloc(m.Type.T, res_len);
        errdefer beam.allocator.free(res_types);
        var res_rest = fields.get("result_types");
        for (res_types) |*rt| {
            rt.* = try decodeType(env, ctx, try beam.get_head_and_iter(env, &res_rest));
        }

        const attr_len = try beam.get_list_length(env, fields.get("attributes"));
        const attrs = try beam.allocator.alloc(c.MlirNamedAttribute, attr_len);
        errdefer beam.allocator.free(attrs);
        var attr_rest = fields.get("attributes");
        for (attrs) |*na| {
            const a_fields = try tuple(env, try beam.get_head_and_iter(env, &attr_rest), 2);
            const a_name = try string_ref.get_binary_as_string_ref(env, a_fields[0]);
            const a_val = try decodeAttribute(env, ctx, a_fields[1]);
            na.* = c.mlirNamedAttributeGet(c.mlirIdentifierGet(ctx, a_name), a_val);
        }

        const succ_len = try beam.get_list_length(env, fields.get("successors"));
        const succs = try beam.allocator.alloc(m.Block.T, succ_len);
        errdefer beam.allocator.free(succs);
        var succ_rest = fields.get("successors");
        for (succs) |*sb| {
            sb.* = try m.Block.resource.fetch(env, try beam.get_head_and_iter(env, &succ_rest));
            if (sb.ptr == null) return error.InvalidBatchSuccessor;
        }

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
        initialized_specs += 1;
    }

    const created_ops = try beam.allocator.alloc(m.Operation.T, count);
    defer beam.allocator.free(created_ops);
    const op_terms = try beam.allocator.alloc(beam.term, count);
    defer beam.allocator.free(op_terms);
    const res_list_terms = try beam.allocator.alloc(beam.term, count);
    defer beam.allocator.free(res_list_terms);

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
        if (resolved_operands.len > 0) {
            c.mlirOperationStateAddOperands(&state, @intCast(resolved_operands.len), resolved_operands.ptr);
        }
        if (spec.result_types.len > 0) {
            c.mlirOperationStateAddResults(&state, @intCast(spec.result_types.len), spec.result_types.ptr);
        }
        if (spec.attributes.len > 0) {
            c.mlirOperationStateAddAttributes(&state, @intCast(spec.attributes.len), spec.attributes.ptr);
        }
        if (spec.successors.len > 0) {
            c.mlirOperationStateAddSuccessors(&state, @intCast(spec.successors.len), spec.successors.ptr);
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
