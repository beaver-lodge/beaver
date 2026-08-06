const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;

const rewrite_pattern = @import("rewrite_pattern.zig");
const resource_sync = @import("resource_sync.zig");

pub const nifs = rewrite_pattern.nifs;

export const rewrite_pattern_nifs: [nifs.len]e.ErlNifFunc = nifs;
export const rewrite_pattern_nifs_len: usize = nifs.len;

export fn rewrite_pattern_open(env: beam.env) void {
    kinda.callback_runtime.ReplyToken.open(env);
    resource_sync.syncResourceTypes();
}
