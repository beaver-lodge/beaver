const std = @import("std");
const builtin = @import("builtin");
const os = builtin.os.tag;

pub fn build(b: *std.Build) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const mix_env = std.process.getEnvVarOwned(allocator, "MIX_ENV") catch "";
    var optimize: std.builtin.OptimizeMode = .ReleaseSafe;
    if (std.mem.eql(u8, mix_env, "prod")) {
        optimize = .ReleaseSafe;
    } else if (std.mem.eql(u8, mix_env, "test")) {
        optimize = .Debug;
    }
    const lib = b.addSharedLibrary(.{ .name = "BeaverNIF", .root_source_file = b.path("src/main.zig"), .optimize = optimize, .target = b.standardTargetOptions(.{}) });
    std.log.info("Building {s}, MIX_ENV={s}, {?}\n", .{ lib.name, mix_env, lib.root_module.optimize });
    const kinda = b.dependency("kinda", .{});
    lib.root_module.addImport("kinda", kinda.module("kinda"));
    lib.root_module.addImport("erl_nif", kinda.module("erl_nif"));
    lib.root_module.addImport("beam", kinda.module("beam"));
    lib.root_module.addIncludePath(.{ .cwd_relative = "include" });
    if (os == .linux) {
        lib.root_module.addRPathSpecial("$ORIGIN");
    }
    if (os == .macos) {
        lib.root_module.addRPathSpecial("@loader_path");
    }
    lib.linkSystemLibrary("MLIRBeaver");
    lib.linker_allow_shlib_undefined = true;
    b.installArtifact(lib);
}
