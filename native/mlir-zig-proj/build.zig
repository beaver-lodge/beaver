const std = @import("std");
const builtin = @import("builtin");
const os = builtin.os.tag;

pub fn build(b: *std.build.Builder) void {
    var lib_name: []const u8 = undefined;
    const lib_name_key = "KINDA_LIB_NAME";
    const des = "required -D" ++ lib_name_key;
    if (b.option([]const u8, lib_name_key, des)) |v| {
        lib_name = v;
    } else @panic(des);
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const target: std.zig.CrossTarget = .{};
    const lib = b.addSharedLibrary(.{
        .name = lib_name,
        .root_source_file = .{ .path = "src/main.zig" },
        .optimize = .Debug,
        .target = target,
    });
    const kinda = b.dependency("kinda", .{});
    lib.addModule("kinda", kinda.module("kinda"));
    lib.addModule("erl_nif", kinda.module("erl_nif"));
    lib.addModule("beam", kinda.module("beam"));
    if (os == .linux) {
        lib.addRPath(.{ .path = "@loader_path" });
    }
    if (os == .macos) {
        lib.addRPath(.{ .path = ":$ORIGIN" });
    }
    lib.linkSystemLibrary("MLIRBeaver");
    lib.linker_allow_shlib_undefined = true;
    b.installArtifact(lib);
}
