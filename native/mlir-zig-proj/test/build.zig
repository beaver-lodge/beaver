const std = @import("std");
const builtin = @import("builtin");
const os = builtin.os.tag;

pub fn build(b: *std.build.Builder) void {
    const env_key = "KINDA_LIB_NAME";
    var lib_name: []u8 = undefined;
    if (std.process.getEnvVarOwned(b.allocator, env_key)) |v| {
        lib_name = v;
    } else |_| {
        @panic("Env var empty: " ++ env_key);
    }
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const target: std.zig.CrossTarget = .{};
    const lib = b.addSharedLibrary(.{
        .name = lib_name,
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
