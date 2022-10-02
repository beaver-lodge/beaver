const std = @import("std");
const kinda = @import("build.imp.zig");

pub fn build(b: *std.build.Builder) void {
    b.cache_root = kinda.cache_root;
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();
    const lib = b.addSharedLibrary("beaver-" ++ kinda.lib_suffix, "src/main.zig", .unversioned);
    lib.setBuildMode(mode);
    lib.addSystemIncludeDir(kinda.erts_include);
    lib.addSystemIncludeDir(kinda.llvm_include);
    lib.addSystemIncludeDir(kinda.beaver_include);
    lib.addLibPath(kinda.beaver_libdir);
    lib.addRPath(kinda.beaver_libdir);
    lib.linkSystemLibrary("MLIRBeaver");
    lib.linker_allow_shlib_undefined = true;
    lib.install();
}
