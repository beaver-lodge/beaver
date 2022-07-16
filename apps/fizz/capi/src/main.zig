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

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
  fizz.open_generated_resource_types(env);
  return 0;
}

const entry = e.ErlNifEntry{
  .major = 2,
  .minor = 16,
  .name = "Elixir.Fizz.MLIR.CAPI",
  .num_of_funcs = fizz.generated_nifs.len,
  .funcs = &(fizz.generated_nifs[0]),
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
  return &entry;
}
