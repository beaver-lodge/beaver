const std = @import("std");
const builtin = @import("builtin");
const os = builtin.os.tag;

pub fn build(b: *std.Build) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const lib = b.addSharedLibrary(.{ .name = "BeaverNIF", .root_source_file = b.path("src/main.zig"), .optimize = .Debug, .target = b.standardTargetOptions(.{}) });
    const kinda = b.dependency("kinda", .{});
    lib.root_module.addImport("kinda", kinda.module("kinda"));
    lib.root_module.addImport("erl_nif", kinda.module("erl_nif"));
    lib.root_module.addImport("beam", kinda.module("beam"));
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
