const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const mlir_capi = @import("mlir_capi.zig");

const NifFunc = e.ErlNifFunc;

// NIF tables exported by the partitioned native libraries. Each partition is
// compiled as its own library so editing one domain does not force the
// comptime-heavy CAPI registry to be regenerated. The tables are concatenated
// at load time below because comptime values cannot cross artifact boundaries.
extern const core_nifs: [0]NifFunc;
extern const core_nifs_len: usize;
extern const conversion_nifs: [0]NifFunc;
extern const conversion_nifs_len: usize;
extern const callback_bridge_nifs: [0]NifFunc;
extern const callback_bridge_nifs_len: usize;
extern const rewrite_pattern_nifs: [0]NifFunc;
extern const rewrite_pattern_nifs_len: usize;

// Registration hooks exported by the partitioned native libraries.
extern fn core_register_all_passes() void;
extern fn core_open_all(env: beam.env) void;
extern fn conversion_open(env: beam.env) void;
extern fn callback_bridge_open(env: beam.env) void;
extern fn rewrite_pattern_open(env: beam.env) void;

const max_nifs = 4096;
var assembled_nifs: [max_nifs]NifFunc = undefined;
var assembled_len: usize = 0;

fn appendNifs(table: [*]const NifFunc, len: usize, index: *usize) void {
    for (table[0..len]) |nif_entry| {
        assembled_nifs[index.*] = nif_entry;
        index.* += 1;
    }
}

fn assembleNifs() []NifFunc {
    const total =
        core_nifs_len +
        conversion_nifs_len +
        callback_bridge_nifs_len +
        rewrite_pattern_nifs_len;
    if (total > max_nifs) @panic("assembled NIF table exceeds static capacity");
    var index: usize = 0;
    appendNifs(@ptrCast(&core_nifs), core_nifs_len, &index);
    appendNifs(@ptrCast(&conversion_nifs), conversion_nifs_len, &index);
    appendNifs(@ptrCast(&callback_bridge_nifs), callback_bridge_nifs_len, &index);
    appendNifs(@ptrCast(&rewrite_pattern_nifs), rewrite_pattern_nifs_len, &index);
    assembled_len = index;
    return assembled_nifs[0..assembled_len];
}

export fn nif_load(env: beam.env, _: [*c]?*anyopaque, _: beam.term) c_int {
    core_register_all_passes();
    core_open_all(env);
    conversion_open(env);
    callback_bridge_open(env);
    rewrite_pattern_open(env);
    return 0;
}

export fn nif_upgrade(
    env: beam.env,
    priv_data: [*c]?*anyopaque,
    _: [*c]?*anyopaque,
    load_info: beam.term,
) c_int {
    return nif_load(env, priv_data, load_info);
}

export fn nif_unload(_: beam.env, _: ?*anyopaque) void {}

var entry: e.ErlNifEntry = undefined;

export fn nif_init() *const e.ErlNifEntry {
    const nifs = assembleNifs();
    entry = e.ErlNifEntry{
        .major = 2,
        .minor = 16,
        .name = mlir_capi.root_module,
        .num_of_funcs = @intCast(nifs.len),
        .funcs = nifs.ptr,
        .load = nif_load,
        .reload = null,
        .upgrade = nif_upgrade,
        .unload = nif_unload,
        .vm_variant = "beam.vanilla",
        .options = 1,
        .sizeof_ErlNifResourceTypeInit = @sizeOf(e.ErlNifResourceTypeInit),
        .min_erts = "erts-13.0",
    };
    return &entry;
}
