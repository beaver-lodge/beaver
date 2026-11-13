//! Named, comptime-specialized views over the shared batch wire schema.
const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const e = kinda.erl_nif;
pub const schema = @import("ir_batch_protocol");
fn index(comptime names: []const []const u8, comptime name: []const u8) usize {
    inline for (names, 0..) |field, i| {
        if (comptime std.mem.eql(u8, field, name)) return i;
    }
    @compileError("unknown batch protocol field: " ++ name);
}
pub const OperandKind = enum(c_int) {
    value = index(schema.operand_kinds, "value"),
    result = index(schema.operand_kinds, "result"),
    block_arg = index(schema.operand_kinds, "block_arg"),
    cfg_block_arg = index(schema.operand_kinds, "cfg_block_arg"),
};
fn TupleView(comptime names: []const []const u8, comptime required: usize) type {
    return struct {
        fields: []const beam.term,
        const Self = @This();
        pub fn decode(env: beam.env, term: beam.term) !Self {
            var count: c_int = 0;
            var fields: [*c]const beam.term = undefined;
            if (e.enif_get_tuple(env, term, &count, &fields) == 0 or count < required or count > names.len) {
                return error.InvalidExecutionArguments;
            }
            return .{ .fields = fields[0..@intCast(count)] };
        }
        pub fn get(self: Self, comptime name: []const u8) beam.term {
            return self.fields[index(names, name)];
        }
        pub fn optional(self: Self, comptime name: []const u8) ?beam.term {
            const i = index(names, name);
            return if (i < self.fields.len) self.fields[i] else null;
        }
    };
}
pub const Operation = TupleView(schema.operation_fields, schema.operation_required);
pub const Operand = TupleView(schema.operand_fields, schema.operand_fields.len);
pub fn operandKind(env: beam.env, term: beam.term) !OperandKind {
    const tag = try beam.get(c_int, env, term);
    inline for (@typeInfo(OperandKind).@"enum".fields) |field| {
        if (tag == field.value) return @field(OperandKind, field.name);
    }
    return error.InvalidBatchOperandReference;
}
