const std = @import("std");
const testing = std.testing;

export fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try testing.expect(add(3, 7) == 10);
}
const beam = @import("beam.zig");
const e = @import("erl_nif.zig");
const fizz = @import("mlir.fizz.gen.zig");
pub const c = fizz.c;

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
  fizz.open_generated_resource_types(env);
  return 0;
}

fn get_all_registered_ops(env: beam.env) !beam.term {
    const ctx = c.mlirContextCreate();
    c.mlirRegisterAllDialects(ctx);
    const num_op: isize = c.beaverGetNumRegisteredOperations(ctx);
    var i: isize = 0;
    var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(usize, num_op));
    while (i < num_op) : ({ i += 1; }) {
        const registered_op_name = c.beaverGetRegisteredOperationName(ctx, i);
        const dialect_name = c.beaverRegisteredOperationNameGetDialectName(registered_op_name);
        const op_name = c.beaverRegisteredOperationNameGetOpName(registered_op_name);
        var tuple_slice: []beam.term = try beam.allocator.alloc(beam.term, 2);
        defer beam.allocator.free(tuple_slice);
        tuple_slice[0] = beam.make_cstring_charlist(env, dialect_name.data);
        tuple_slice[1] = beam.make_cstring_charlist(env, op_name.data);
        ret[@intCast(usize, i)] = beam.make_tuple(env, tuple_slice);
    }
    c.mlirContextDestroy(ctx);
    return beam.make_term_list(env, ret);
}
export fn registered_ops(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
    return get_all_registered_ops(env) catch beam.make_error_binary(env, "launching nif");
}

export fn cstring_to_charlist(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]const u8 = undefined; arg0 = beam.fetch_resource(arg0, env, fizz.resource_type__c_ptr_const_u8, args[0]);
  return beam.make_cstring_charlist(env, arg0);
}

pub export const handwritten_nifs = ([_]e.ErlNifFunc{
  e.ErlNifFunc{.name = "registered_ops", .arity = 0, .fptr = registered_ops, .flags = 0},
  e.ErlNifFunc{.name = "cstring_to_charlist", .arity = 1, .fptr = cstring_to_charlist, .flags = 0},
});

pub export const num_nifs = fizz.generated_nifs.len + handwritten_nifs.len;
pub export var nifs: [num_nifs]e.ErlNifFunc = undefined;

const entry = e.ErlNifEntry{
  .major = 2,
  .minor = 16,
  .name = "Elixir.Beaver.MLIR.CAPI",
  .num_of_funcs = num_nifs,
  .funcs = &(nifs[0]),
  .load = nif_load,
  .reload = null,   // currently unsupported
  .upgrade = null,  // currently unsupported
  .unload = null,   // currently unsupported
  .vm_variant = "beam.vanilla",
  .options = 1,
  .sizeof_ErlNifResourceTypeInit = @sizeOf(e.ErlNifResourceTypeInit),
  .min_erts = "erts-13.0"
};

export fn nif_init() *const e.ErlNifEntry{
  var i: usize = 0;
  while (i < fizz.generated_nifs.len) : ({ i += 1; }) {
      nifs[i] = fizz.generated_nifs[i];
  }
  var j: usize = 0;
  while (j < handwritten_nifs.len) : ({ j += 1; }) {
      nifs[fizz.generated_nifs.len + j] = handwritten_nifs[j];
  }
  return &entry;
}
