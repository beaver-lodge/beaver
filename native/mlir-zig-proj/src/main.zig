const std = @import("std");
const mem = @import("std").mem;
const testing = std.testing;
const beam = @import("beam");
const kinda = @import("kinda");
const e = @import("erl_nif");
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const enif_support = @import("enif_support.zig");
const diagnostic = @import("diagnostic.zig");
const pass = @import("pass.zig");
const registry = @import("registry.zig");
const pointer = @import("pointer.zig");
const string_ref = @import("string_ref.zig");
const Printer = string_ref.Printer;
const memref = @import("memref.zig");

const handwritten_nifs = @import("wrapper.zig").nif_entries ++ mlir_capi.EntriesOfKinds ++ pass.nifs ++ registry.nifs ++ string_ref.nifs ++ diagnostic.nifs ++ pointer.nifs ++ memref.nifs ++ pointer.PtrOwner.Kind.nifs ++ .{
    e.ErlNifFunc{ .name = "beaver_raw_to_string_attribute", .arity = 1, .fptr = Printer(mlir_capi.Attribute, c.mlirAttributePrint).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_type", .arity = 1, .fptr = Printer(mlir_capi.Type, c.mlirTypePrint).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_operation", .arity = 1, .fptr = Printer(mlir_capi.Operation, c.mlirOperationPrint).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_operation_specialized", .arity = 1, .fptr = Printer(mlir_capi.Operation, c.beaverOperationPrintSpecializedFrom).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_operation_generic", .arity = 1, .fptr = Printer(mlir_capi.Operation, c.beaverOperationPrintGenericOpForm).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_operation_bytecode", .arity = 1, .fptr = Printer(mlir_capi.Operation, c.mlirOperationWriteBytecode).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_value", .arity = 1, .fptr = Printer(mlir_capi.Value, c.mlirValuePrint).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_pm", .arity = 1, .fptr = Printer(mlir_capi.OpPassManager, c.mlirPrintPassPipeline).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_affine_map", .arity = 1, .fptr = Printer(mlir_capi.AffineMap, c.mlirAffineMapPrint).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_to_string_location", .arity = 1, .fptr = Printer(mlir_capi.Location, c.beaverLocationPrint).to_string, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_jit_invoke_with_terms", .arity = 3, .fptr = enif_support.beaver_raw_jit_invoke_with_terms, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_jit_register_enif", .arity = 1, .fptr = enif_support.beaver_raw_jit_register_enif, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_enif_signatures", .arity = 1, .fptr = enif_support.beaver_raw_enif_signatures, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_enif_functions", .arity = 0, .fptr = enif_support.beaver_raw_enif_functions, .flags = 0 },
    e.ErlNifFunc{ .name = "beaver_raw_mlir_type_of_enif_obj", .arity = 2, .fptr = enif_support.beaver_raw_mlir_type_of_enif_obj, .flags = 0 },
};

const num_nifs = handwritten_nifs.len;
export var nifs: [num_nifs]e.ErlNifFunc = handwritten_nifs;

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
    kinda.open_internal_resource_types(env);
    mlir_capi.open_generated_resource_types(env);
    memref.open_all(env);
    beam.open_resource_wrapped(env, pass.Token);
    kinda.Internal.OpaqueStruct.open_all(env);
    pointer.PtrOwner.Kind.open(env);
    return 0;
}

const entry = e.ErlNifEntry{
    .major = 2,
    .minor = 16,
    .name = mlir_capi.root_module,
    .num_of_funcs = num_nifs,
    .funcs = &(nifs[0]),
    .load = nif_load,
    .reload = null, // currently unsupported
    .upgrade = null, // currently unsupported
    .unload = null, // currently unsupported
    .vm_variant = "beam.vanilla",
    .options = 1,
    .sizeof_ErlNifResourceTypeInit = @sizeOf(e.ErlNifResourceTypeInit),
    .min_erts = "erts-13.0",
};

export fn nif_init() *const e.ErlNifEntry {
    return &entry;
}
