const std = @import("std");
const builtin = @import("builtin");
const os = builtin.os.tag;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .name = "BeaverNIF",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    std.log.info("Building {s}, {any}\n", .{ lib.name, lib.root_module.optimize });

    const kinda = b.dependency("kinda", .{});
    lib.root_module.addImport("kinda", kinda.module("kinda"));
    lib.root_module.addIncludePath(.{ .cwd_relative = "include" });

    if (os == .linux) {
        lib.root_module.addRPathSpecial("$ORIGIN");
        lib.linkLibC();
    }
    if (os == .macos) {
        lib.root_module.addRPathSpecial("@loader_path");
    }

    lib.linkSystemLibrary("MLIRBeaver");
    lib.linker_allow_shlib_undefined = true;
    b.installArtifact(lib);
}
