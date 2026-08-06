const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const prelude = @import("prelude.zig");
const mlir_capi = @import("mlir_capi.zig");

extern fn beaverContextRegisterTritonDialects(context: mlir_capi.Context.T) bool;
extern fn beaverRegisterTritonPasses() bool;

pub fn triton_register_dialects(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const context = try mlir_capi.Context.resource.fetch(env, args[0]);
    return beam.make_atom(env, if (beaverContextRegisterTritonDialects(context)) "true" else "false");
}

pub fn triton_register_passes(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    return beam.make_atom(env, if (beaverRegisterTritonPasses()) "true" else "false");
}

pub const nifs = .{
    prelude.beaverRawNIF(@This(), "triton_register_dialects", 1),
    prelude.beaverRawNIF(@This(), "triton_register_passes", 0),
};
