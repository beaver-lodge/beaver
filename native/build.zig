const std = @import("std");
const builtin = @import("builtin");
const os = builtin.os.tag;

fn generateWrapper(b: *std.Build, generated_dir: []const u8, mlir_include_dir: []const u8) void {
    // Create include directory
    _ = b.run(&.{
        "mkdir", "-p", b.pathJoin(&.{ generated_dir, "include", "mlir-c", "Beaver" }),
    });

    // Generate header file
    _ = b.run(&.{
        "elixir",             "tools/wrapper/gen_header.exs",
        "--mlir-include-dir", mlir_include_dir,
        "--output",           b.pathJoin(&.{ generated_dir, "include", "mlir-c", "Beaver", "wrapper.h" }),
    });

    // Generate wrapper files using clang AST
    _ = b.run(&.{
        "sh", "-c", b.fmt(
            \\zig cc -E -Xclang -ast-dump=json "{s}/include/mlir-c/Beaver/wrapper.h" \
            \\  -I include -I "{s}" \
            \\| elixir tools/wrapper/gen_stub.exs --elixir "{s}/capi_functions.ex" --zig "{s}/wrapper.zig"
        , .{ generated_dir, mlir_include_dir, b.install_path, generated_dir }),
    });
}

fn createCMakeStep(b: *std.Build, llvm_cmake_dir: []const u8, mlir_cmake_dir: []const u8, optimize: std.builtin.OptimizeMode) *std.Build.Step {
    const step = b.step("cmake", "Build and install CMake targets");

    const cmake_build_dir = b.pathJoin(&.{ b.install_path, "cmake_build" });
    const cmake_cache_path = b.pathJoin(&.{ cmake_build_dir, "CMakeCache.txt" });

    // cmake_build command
    const cmake_configure = b.addSystemCommand(&.{ "cmake", "-G", "Ninja", "-B", cmake_build_dir });
    cmake_configure.addArgs(&.{
        b.fmt("-DLLVM_DIR={s}", .{llvm_cmake_dir}),
        b.fmt("-DMLIR_DIR={s}", .{mlir_cmake_dir}),
        b.fmt("-DCMAKE_INSTALL_PREFIX={s}", .{b.install_path}),
        b.fmt("-DCMAKE_INSTALL_MESSAGE={s}", .{"LAZY"}),
        b.fmt("-DCMAKE_BUILD_TYPE={s}", .{switch (optimize) {
            .Debug => "Debug",
            .ReleaseSafe => "RelWithDebInfo",
            .ReleaseFast => "Release",
            .ReleaseSmall => "MinSizeRel",
        }}),
    });
    const cmake_build_install = b.addSystemCommand(&.{
        "cmake", "--build", cmake_build_dir, "--target", "install", "--" , "--quiet"
    });
    step.dependOn(&cmake_build_install.step);

    std.fs.accessAbsolute(cmake_cache_path, .{}) catch {
        step.dependOn(&cmake_configure.step);
        cmake_build_install.step.dependOn(&cmake_configure.step);
    };
    return step;
}

/// Creates ODS extraction steps. Returns both a container step for manual execution
/// and the final extraction step for proper dependency management.
///
/// Important: The container_step does not depend on cmake. Depending on the container_step
/// instead of the extraction_step will lead to premature execution since the dependency
/// chain is broken.
fn createODSExtractionStep(b: *std.Build, generated_dir: []const u8, mlir_include_dir: []const u8, cmake_step: *std.Build.Step) *std.Build.Step {
    const step = b.step("ods_extraction", "Extract ODS information and convert to inspection format");
    const create_include_dir = b.addSystemCommand(&.{
        "mkdir", "-p", generated_dir,
    });

    const dump_ods = b.addSystemCommand(&.{
        "sh",
        "-c",
        b.fmt(
            \\elixir tools/ods-extract/dump_ods.exs --mlir-include-dir "{s}" \
            \\| xargs "{s}/tools/ods-extract" \
            \\   -I "{s}" \
            \\   -I "{s}/mlir/Dialect/ArmSME/IR" \
            \\   -I "{s}/mlir/Dialect/IRDL/IR" \
            \\   -I "{s}/mlir/Dialect/UB/IR" \
            \\   -write-if-changed -o "{s}/ods_dump.json"
        , .{ mlir_include_dir, b.install_path, mlir_include_dir, mlir_include_dir, mlir_include_dir, mlir_include_dir, generated_dir }),
    });
    dump_ods.step.dependOn(cmake_step);
    dump_ods.step.dependOn(&create_include_dir.step);
    step.dependOn(&dump_ods.step);

    const json_to_inspection = b.addSystemCommand(&.{
        "elixir",   "tools/json_to_inspection.exs",
        "--input",  b.pathJoin(&.{ generated_dir, "ods_dump.json" }),
        "--output", b.pathJoin(&.{ b.install_path, "ods_dump.ex" }),
    });
    json_to_inspection.step.dependOn(&dump_ods.step);
    step.dependOn(&json_to_inspection.step);

    return step;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    var optimize = b.standardOptimizeOption(.{});
    if (std.posix.getenv("MIX_ENV")) |env| {
        if (std.mem.eql(u8, env, "test")) {
            optimize = .Debug;
        } else if (std.mem.eql(u8, env, "dev") or std.mem.eql(u8, env, "prod")) {
            optimize = .ReleaseSafe;
        }
    }

    // Environment variables and paths
    const llvm_config_path = b.option([]const u8, "llvm-config", "Path to llvm-config") orelse std.posix.getenv("LLVM_CONFIG_PATH") orelse "llvm-config";
    const generated_dir = b.pathJoin(&.{ b.install_path, "generated" });

    const llvm_lib_dir_raw = b.run(&.{ llvm_config_path, "--libdir" });
    const llvm_lib_dir = std.mem.trim(u8, llvm_lib_dir_raw, " \t\n\r");
    const llvm_cmake_dir = b.pathJoin(&.{ llvm_lib_dir, "cmake", "llvm" });
    const mlir_cmake_dir = b.pathJoin(&.{ llvm_lib_dir, "cmake", "mlir" });
    const llvm_install_dir = b.pathResolve(&.{ llvm_lib_dir, ".." });
    const mlir_include_dir = b.pathJoin(&.{ llvm_install_dir, "include" });
    std.log.info("Using LLVM installation: {s}", .{llvm_install_dir});

    b.addSearchPrefix(llvm_install_dir);
    // add install path to search prefixes because CMake will shared this path as its install prefix
    b.addSearchPrefix(b.install_path);

    generateWrapper(b, generated_dir, mlir_include_dir);
    const cmake_step = createCMakeStep(b, llvm_cmake_dir, mlir_cmake_dir, optimize);
    // ODS extraction step (depends on cmake_build)
    const ods_extraction_step = createODSExtractionStep(b, generated_dir, mlir_include_dir, cmake_step);
    b.getInstallStep().dependOn(ods_extraction_step);

    // Default target
    const lib = b.addLibrary(.{
        .name = "BeaverNIF",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib.step.dependOn(cmake_step);

    std.log.info("Setting optimization mode for {s}: {any}", .{ lib.name, lib.root_module.optimize });

    const kinda = b.dependency("kinda", .{});
    lib.root_module.addImport("kinda", kinda.module("kinda"));
    lib.root_module.addIncludePath(.{ .cwd_relative = "include" });
    // add these to get ZLS working properly
    lib.root_module.addIncludePath(.{ .cwd_relative = mlir_include_dir });
    lib.root_module.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ generated_dir, "include" }) });
    // add generated wrapper.zig as a module
    const wrapper_module = b.createModule(.{
        .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ generated_dir, "wrapper.zig" }) },
        .target = target,
        .optimize = optimize,
    });
    wrapper_module.addImport("kinda", kinda.module("kinda"));
    lib.root_module.addImport("wrapper", wrapper_module);

    if (os == .linux) {
        lib.root_module.addRPathSpecial("$ORIGIN");
        lib.linkLibC();
    }
    if (os == .macos) {
        lib.root_module.addRPathSpecial("@loader_path");
    }
    lib.linkSystemLibrary("MLIRBeaver");
    lib.linker_allow_shlib_undefined = true;
    // copy runtime libs
    b.installDirectory(.{ .source_dir = .{ .cwd_relative = llvm_lib_dir }, .install_dir = .prefix, .install_subdir = "lib", .include_extensions = &.{ ".so", ".dylib", ".dll" } });
    b.installArtifact(lib);
    const check = b.step("check", "Check if compiles");
    check.dependOn(b.getInstallStep());
}
