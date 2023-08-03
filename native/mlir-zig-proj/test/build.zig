const std = @import("std");
const kinda = @import("build.imp.zig");
const builtin = @import("builtin");
const os = builtin.os.tag;

pub fn build(b: *std.build.Builder) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const target: std.zig.CrossTarget = .{};
    const lib = b.addSharedLibrary(.{
        .name = kinda.lib_name,
        .root_source_file = .{ .path = "src/main.zig" },
        .optimize = .Debug,
        .target = target,
    });
    if (os == .linux) {
        lib.addRPath("$ORIGIN");
    }
    if (os == .macos) {
        lib.addRPath("@loader_path");
    }
    lib.linkSystemLibrary("MLIRBeaver");
    lib.linker_allow_shlib_undefined = true;
    b.installArtifact(lib);
}
