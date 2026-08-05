const std = @import("std");
const builtin = @import("builtin");
const os = builtin.os.tag;

fn resolveLlvmConfigPath(b: *std.Build) []const u8 {
    return b.option([]const u8, "llvm-config", "Path to llvm-config") orelse
        b.graph.environ_map.get("LLVM_CONFIG_PATH") orelse
        b.findProgram(&.{"llvm-config"}, &.{
            b.pathResolve(&.{ "priv", "llvm-prebuilt", "bin" }),
            "/opt/homebrew/opt/llvm/bin",
            "/usr/local/opt/llvm/bin",
            "/opt/local/libexec/llvm/bin",
        }) catch "llvm-config";
}

fn createCapiModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, mlir_include_dir: []const u8) *std.Build.Module {
    const header_generator = b.addSystemCommand(&.{"elixir"});
    // Header discovery must observe additions and removals in both include trees.
    header_generator.has_side_effects = true;
    header_generator.addFileArg(b.path("native/tools/capi/gen_header.exs"));
    header_generator.addArg("--mlir-include-dir");
    header_generator.addDirectoryArg(.{ .cwd_relative = mlir_include_dir });
    header_generator.addArg("--beaver-include-dir");
    header_generator.addDirectoryArg(b.path("native/include"));
    header_generator.addArg("--output");
    const capi_header = header_generator.addOutputFileArg("beaver_capi.h");

    const ast_dump = b.addSystemCommand(&.{ b.graph.zig_exe, "cc", "-E", "-Xclang", "-ast-dump=json", "-x", "c-header" });
    ast_dump.addArg("-I");
    ast_dump.addDirectoryArg(b.path("native/include"));
    ast_dump.addArg("-I");
    ast_dump.addDirectoryArg(.{ .cwd_relative = mlir_include_dir });
    ast_dump.addArgs(&.{ "-MD", "-MF" });
    _ = ast_dump.addDepFileOutputArg("capi_ast.d");
    ast_dump.addFileArg(capi_header);
    const capi_ast = ast_dump.captureStdOut(.{ .basename = "capi_ast.json" });

    const manifest_generator = b.addSystemCommand(&.{"elixir"});
    if (b.graph.environ_map.get("JASON_EBIN_PATH")) |jason_ebin_path| {
        manifest_generator.addArgs(&.{ "-pa", jason_ebin_path });
    }
    manifest_generator.addFileArg(b.path("native/tools/capi/gen_manifest.exs"));
    manifest_generator.addArg("--declaration");
    const declaration_manifest = manifest_generator.addOutputFileArg("capi_manifest.json");
    manifest_generator.addArg("--callback-bridge");
    const callback_bridge_manifest = manifest_generator.addOutputFileArg("capi_callback_bridge.json");
    manifest_generator.setStdIn(.{ .lazy_path = capi_ast });

    b.getInstallStep().dependOn(&b.addInstallFile(declaration_manifest, "capi_manifest.json").step);
    b.getInstallStep().dependOn(&b.addInstallFile(callback_bridge_manifest, "capi_callback_bridge.json").step);

    const translate_c = b.addTranslateC(.{
        .root_source_file = capi_header,
        .target = target,
        .optimize = optimize,
    });
    translate_c.defineCMacro("_NO_CRT_STDIO_INLINE", "1");
    translate_c.addIncludePath(b.path("native/include"));
    translate_c.addIncludePath(.{ .cwd_relative = mlir_include_dir });
    return translate_c.createModule();
}

fn createCMakeStep(b: *std.Build, llvm_cmake_dir: []const u8, mlir_cmake_dir: []const u8, optimize: std.builtin.OptimizeMode) *std.Build.Step {
    const step = b.step("cmake", "Build and install CMake targets");

    const cmake_build_dir = b.pathJoin(&.{ b.install_path, "cmake_build" });
    const cmake_cache_path = b.pathJoin(&.{ cmake_build_dir, "CMakeCache.txt" });

    // cmake_build command
    const cmake_configure = b.addSystemCommand(&.{ "cmake", "-S", "native", "-G", "Ninja", "-B", cmake_build_dir });
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
    const cmake_build_install = b.addSystemCommand(&.{ "cmake", "--build", cmake_build_dir, "--target", "install" });
    step.dependOn(&cmake_build_install.step);

    std.Io.Dir.accessAbsolute(b.graph.io, cmake_cache_path, .{}) catch {
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
    const step = b.step("ods_extraction", "Extract ODS information as JSON");
    const create_include_dir = b.addSystemCommand(&.{
        "mkdir", "-p", generated_dir,
    });

    const dump_ods = b.addSystemCommand(&.{
        "sh",
        "-c",
        b.fmt(
            \\elixir native/tools/ods-extract/dump_ods.exs --mlir-include-dir "{s}" \
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

    return step;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    var optimize = b.standardOptimizeOption(.{});
    if (b.graph.environ_map.get("MIX_ENV")) |env| {
        if (std.mem.eql(u8, env, "test")) {
            optimize = .Debug;
        } else if (std.mem.eql(u8, env, "dev") or std.mem.eql(u8, env, "prod")) {
            optimize = .ReleaseSafe;
        }
    }

    // Environment variables and paths
    const llvm_config_path = resolveLlvmConfigPath(b);
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

    const capi_module = createCapiModule(b, target, optimize, mlir_include_dir);
    const cmake_step = createCMakeStep(b, llvm_cmake_dir, mlir_cmake_dir, optimize);
    // ODS extraction step (depends on cmake_build)
    const ods_extraction_step = createODSExtractionStep(b, generated_dir, mlir_include_dir, cmake_step);
    b.getInstallStep().dependOn(ods_extraction_step);

    // Default target
    const lib = b.addLibrary(.{
        .name = "BeaverNIF",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("native/src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lib.step.dependOn(cmake_step);

    std.log.info("Setting optimization mode for {s}: {any}", .{ lib.name, lib.root_module.optimize });

    const kinda = b.dependency("kinda", .{});
    lib.root_module.addImport("kinda", kinda.module("kinda"));
    lib.root_module.addImport("c_api", capi_module);
    lib.root_module.addIncludePath(.{ .cwd_relative = "native/include" });
    // add these to get ZLS working properly
    lib.root_module.addIncludePath(.{ .cwd_relative = mlir_include_dir });
    if (os == .linux) {
        lib.root_module.addRPathSpecial("$ORIGIN");
        lib.root_module.link_libc = true;
    }
    if (os == .macos) {
        lib.root_module.addRPathSpecial("@loader_path");
        lib.root_module.link_libc = true;
    }
    lib.root_module.linkSystemLibrary("MLIRBeaver", .{ .use_pkg_config = .no });
    lib.linker_allow_shlib_undefined = true;
    // copy runtime libs
    b.installDirectory(.{ .source_dir = .{ .cwd_relative = llvm_lib_dir }, .install_dir = .prefix, .install_subdir = "lib", .include_extensions = &.{ ".so", ".dylib", ".dll" } });
    b.installArtifact(lib);
    const check = b.step("check", "Check if compiles");
    check.dependOn(b.getInstallStep());
}
