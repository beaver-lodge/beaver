const std = @import("std");
const kinda = @import("build.imp.zig");
const builtin = @import("builtin");
const os = builtin.os.tag;

pub fn build(b: *std.build.Builder) void {
    b.cache_root = kinda.cache_root;
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();
    const lib = b.addSharedLibrary(kinda.lib_name, "src/main.zig", .unversioned);
    lib.setBuildMode(mode);
    lib.addSystemIncludePath(kinda.erts_include);
    lib.addSystemIncludePath(kinda.llvm_include);
    lib.addSystemIncludePath(kinda.beaver_include);
    lib.addLibraryPath(kinda.beaver_libdir);
    if (os == .linux) {
        lib.addRPath("$ORIGIN");
    }
    if (os == .macos) {
        lib.addRPath("@loader_path");
    }
    lib.linkSystemLibrary("MLIRBeaver");
    lib.linker_allow_shlib_undefined = true;
    lib.install();
}
