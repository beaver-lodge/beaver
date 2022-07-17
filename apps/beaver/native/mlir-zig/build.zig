const std = @import("std");
const fizz = @import("src/build.fizz.gen.zig");

pub fn build(b: *std.build.Builder) void {
    b.dest_dir = fizz.dest_dir;
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    const lib = b.addSharedLibrary("BeaverCAPI", "src/main.zig", b.version(0, 0, 1));
    lib.setBuildMode(mode);

    // TODO: import from a generated file to include erlang headers
    lib.addSystemIncludeDir("/opt/homebrew/Cellar/erlang/25.0/lib/erlang/erts-13.0/include");
    lib.addSystemIncludeDir(fizz.llvm_include);
    lib.addSystemIncludeDir(fizz.beaver_include);
    lib.addSystemIncludeDir("src");
    lib.addLibPath(b.pathFromRoot("zig-out/lib"));
    lib.linkSystemLibrary("beaver");
    lib.linker_allow_shlib_undefined = true;
    lib.addRPath(b.pathFromRoot("zig-out/lib"));

    lib.install();

    const main_tests = b.addTest("src/main.zig");
    main_tests.setBuildMode(mode);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&main_tests.step);

}
