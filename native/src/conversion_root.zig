const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;

const conversion = @import("conversion.zig");
const resource_sync = @import("resource_sync.zig");

pub const nifs = conversion.nifs;

export const conversion_nifs: [nifs.len]e.ErlNifFunc = nifs;
export const conversion_nifs_len: usize = nifs.len;

export fn conversion_open(env: beam.env) void {
    // Opens this DSO's private callback-runtime library pin; the shared
    // ReplyToken handle is overwritten by the sync below.
    kinda.callback_runtime.ReplyToken.open(env);
    resource_sync.syncResourceTypes();
    conversion.open(env);
}
