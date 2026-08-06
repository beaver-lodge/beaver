const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;

const callback_bridge = @import("callback_bridge.zig");
const resource_sync = @import("resource_sync.zig");

pub const nifs = callback_bridge.nifs;

export const callback_bridge_nifs: [nifs.len]e.ErlNifFunc = nifs;
export const callback_bridge_nifs_len: usize = nifs.len;

export fn callback_bridge_open(env: beam.env) void {
    kinda.callback_runtime.ReplyToken.open(env);
    resource_sync.syncResourceTypes();
    callback_bridge.open(env);
}
