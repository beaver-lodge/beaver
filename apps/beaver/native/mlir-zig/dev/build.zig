const std = @import("std");
const fizz = @import("build.imp.zig");

pub fn build(b: *std.build.Builder) void {
    b.cache_root = fizz.cache_root;
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();
    const lib = b.addSharedLibrary("BeaverNIF", "src/main.zig", b.version(0, 0, 1));
    lib.setBuildMode(mode);
    lib.addSystemIncludeDir(fizz.erts_include);
    lib.addSystemIncludeDir(fizz.llvm_include);
    lib.addSystemIncludeDir(fizz.beaver_include);
    lib.addLibPath(fizz.beaver_libdir);
    lib.addRPath(fizz.beaver_libdir);
    lib.linkSystemLibrary("MLIRBeaver");
    lib.linker_allow_shlib_undefined = true;
    lib.install();
}
