//! A minimal Zig CUDA runner.
//!
//! Loads the CUDA Driver API (`libcuda`) with `dlopen` so the NIF does not
//! link against NVIDIA libraries at build time: on machines without a driver
//! the partition degrades gracefully instead of failing to load. This is the
//! first slice of the Zig CUDA runner that replaces the MLIR C++ CUDA runtime
//! (`libmlir_cuda_runtime.so`), which is absent from eudsl prebuilts.
//!
//! The Elixir side extracts PTX assembly from `gpu.binary` (see
//! `Beaver.MLIR.CUDA.load_gpu_binary/1`) and hands it to `cuda_module_load`;
//! this partition owns driver loading, module/function handles, memory
//! management, copies and kernel launches.

const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const prelude = @import("prelude.zig");
const mutex = @import("mutex.zig");

const CudaResult = c_int;
const CuDevice = *anyopaque;

const cuInitFn = *const fn (flags: c_uint) callconv(.c) CudaResult;
const cuDeviceGetCountFn = *const fn (count: *c_int) callconv(.c) CudaResult;
const cuDeviceGetFn = *const fn (device: *CuDevice, ordinal: c_int) callconv(.c) CudaResult;
const cuDeviceGetNameFn = *const fn (name: [*c]u8, len: c_int, dev: CuDevice) callconv(.c) CudaResult;
const cuModuleLoadDataFn = *const fn (module: *?*anyopaque, data: *const anyopaque) callconv(.c) CudaResult;
const cuModuleGetFunctionFn = *const fn (function: *?*anyopaque, module: ?*anyopaque, name: [*:0]const u8) callconv(.c) CudaResult;
const cuModuleUnloadFn = *const fn (module: ?*anyopaque) callconv(.c) CudaResult;
const cuMemAllocFn = *const fn (ptr: *usize, size: usize) callconv(.c) CudaResult;
const cuMemFreeFn = *const fn (ptr: usize) callconv(.c) CudaResult;
const cuMemcpyFn = *const fn (dst: usize, src: usize, size: usize) callconv(.c) CudaResult;
const cuDevicePrimaryCtxRetainFn = *const fn (pctx: *?*anyopaque, dev: CuDevice) callconv(.c) CudaResult;
const cuCtxSetCurrentFn = *const fn (ctx: ?*anyopaque) callconv(.c) CudaResult;
const cuLaunchKernelFn = *const fn (
    function: ?*anyopaque,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_mem: c_uint,
    stream: ?*anyopaque,
    params: [*c]?*anyopaque,
    extra: [*c]?*anyopaque,
) callconv(.c) CudaResult;

const CudaDriver = struct {
    dynlib: std.DynLib,
    cuInit: cuInitFn,
    cuDeviceGetCount: cuDeviceGetCountFn,
    cuDeviceGet: cuDeviceGetFn,
    cuDeviceGetName: cuDeviceGetNameFn,
    cuModuleLoadData: cuModuleLoadDataFn,
    cuModuleGetFunction: cuModuleGetFunctionFn,
    cuModuleUnload: cuModuleUnloadFn,
    cuMemAlloc: cuMemAllocFn,
    cuMemFree: cuMemFreeFn,
    cuMemcpy: cuMemcpyFn,
    cuDevicePrimaryCtxRetain: cuDevicePrimaryCtxRetainFn,
    cuCtxSetCurrent: cuCtxSetCurrentFn,
    cuLaunchKernel: cuLaunchKernelFn,
    /// Result of `cuInit(0)`, cached so repeated NIF calls do not re-initialize.
    init_result: CudaResult,
    /// Retained primary context, mirroring the C++ runner's `ScopedContext`.
    primary_ctx: ?*anyopaque,
};

/// Candidate names/locations for the CUDA driver shared library. `dlopen`
/// searches the loader path for bare names; absolute paths cover WSL and
/// toolkit installs where the loader path is not configured.
const libcuda_candidates = [_][]const u8{
    "libcuda.so.1",
    "libcuda.so",
    "/usr/lib/wsl/lib/libcuda.so.1",
    "/usr/local/cuda/lib64/libcuda.so.1",
};

var driver: ?CudaDriver = null;
var driver_probe_done: bool = false;
var driver_mutex: mutex.Mutex = .{};

fn loadDriver() ?*const CudaDriver {
    driver_mutex.lock();
    defer driver_mutex.unlock();

    if (driver_probe_done) {
        return if (driver) |*d| d else null;
    }
    driver_probe_done = true;

    for (libcuda_candidates) |path| {
        var lib = std.DynLib.open(path) catch continue;
        const cuInit = lib.lookup(cuInitFn, "cuInit") orelse continue;
        const cuDeviceGetCount = lib.lookup(cuDeviceGetCountFn, "cuDeviceGetCount") orelse continue;
        const cuDeviceGet = lib.lookup(cuDeviceGetFn, "cuDeviceGet") orelse continue;
        const cuDeviceGetName = lib.lookup(cuDeviceGetNameFn, "cuDeviceGetName") orelse continue;
        const cuModuleLoadData = lib.lookup(cuModuleLoadDataFn, "cuModuleLoadData") orelse continue;
        const cuModuleGetFunction = lib.lookup(cuModuleGetFunctionFn, "cuModuleGetFunction") orelse continue;
        const cuModuleUnload = lib.lookup(cuModuleUnloadFn, "cuModuleUnload") orelse continue;
        // CUDA 12+ drivers version the driver API entry points. On CUDA 13.x
        // the plain `cuMemAlloc`/`cuMemFree`/`cuMemcpy` symbols resolve to the
        // legacy entry points that fail with CUDA_ERROR_INVALID_CONTEXT, so
        // prefer the `_v2` variants.
        const cuMemAlloc = lib.lookup(cuMemAllocFn, "cuMemAlloc_v2") orelse lib.lookup(cuMemAllocFn, "cuMemAlloc") orelse continue;
        const cuMemFree = lib.lookup(cuMemFreeFn, "cuMemFree_v2") orelse lib.lookup(cuMemFreeFn, "cuMemFree") orelse continue;
        const cuMemcpy = lib.lookup(cuMemcpyFn, "cuMemcpy_v2") orelse lib.lookup(cuMemcpyFn, "cuMemcpy") orelse continue;
        const cuDevicePrimaryCtxRetain = lib.lookup(cuDevicePrimaryCtxRetainFn, "cuDevicePrimaryCtxRetain") orelse continue;
        const cuCtxSetCurrent = lib.lookup(cuCtxSetCurrentFn, "cuCtxSetCurrent") orelse continue;
        const cuLaunchKernel = lib.lookup(cuLaunchKernelFn, "cuLaunchKernel") orelse continue;

        const init_result = cuInit(0);
        var primary_ctx: ?*anyopaque = null;
        if (init_result == 0) {
            var device: CuDevice = undefined;
            const get_res = cuDeviceGet(&device, 0);
            const retain_res = cuDevicePrimaryCtxRetain(&primary_ctx, device);
            const set_res = cuCtxSetCurrent(primary_ctx);
            _ = get_res;
            _ = retain_res;
            _ = set_res;
        }
        driver = .{
            .dynlib = lib,
            .cuInit = cuInit,
            .cuDeviceGetCount = cuDeviceGetCount,
            .cuDeviceGet = cuDeviceGet,
            .cuDeviceGetName = cuDeviceGetName,
            .cuModuleLoadData = cuModuleLoadData,
            .cuModuleGetFunction = cuModuleGetFunction,
            .cuModuleUnload = cuModuleUnload,
            .cuMemAlloc = cuMemAlloc,
            .cuMemFree = cuMemFree,
            .cuMemcpy = cuMemcpy,
            .cuDevicePrimaryCtxRetain = cuDevicePrimaryCtxRetain,
            .cuCtxSetCurrent = cuCtxSetCurrent,
            .cuLaunchKernel = cuLaunchKernel,
            .init_result = init_result,
            .primary_ctx = primary_ctx,
        };
        return &driver.?;
    }
    return null;
}

fn cuResultError(env: beam.env, name: []const u8, result: CudaResult) beam.term {
    var buf: [160]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "{s} failed with CUDA error {d}", .{ name, result }) catch
        "CUDA call failed";
    return beam.make_error_binary(env, msg);
}

/// Returns `true` when a CUDA driver is loadable and initializable.
pub fn cuda_available(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    const d = loadDriver() orelse return beam.make_bool(env, false);
    return beam.make_bool(env, d.init_result == 0);
}

/// Returns `{:ok, count}` with the number of CUDA devices, or `{:error, reason}`.
pub fn cuda_device_count(env: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
    const d = loadDriver() orelse return beam.make_error_binary(env, "libcuda not found");
    if (d.init_result != 0) return cuResultError(env, "cuInit", d.init_result);

    var count: c_int = 0;
    const result = d.cuDeviceGetCount(&count);
    if (result != 0) return cuResultError(env, "cuDeviceGetCount", result);
    return beam.make_ok_term(env, try beam.make(i32, env, @intCast(count)));
}

/// Returns `{:ok, name}` for the device at `ordinal`, or `{:error, reason}`.
pub fn cuda_device_name(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = loadDriver() orelse return beam.make_error_binary(env, "libcuda not found");
    if (d.init_result != 0) return cuResultError(env, "cuInit", d.init_result);

    const ordinal = try beam.get_i32(env, args[0]);
    var device: CuDevice = undefined;
    const get_result = d.cuDeviceGet(&device, ordinal);
    if (get_result != 0) return cuResultError(env, "cuDeviceGet", get_result);

    var name_buf: [256]u8 = undefined;
    const name_result = d.cuDeviceGetName(&name_buf, @intCast(name_buf.len), device);
    if (name_result != 0) return cuResultError(env, "cuDeviceGetName", name_result);

    const name_len = std.mem.indexOfScalar(u8, &name_buf, 0) orelse name_buf.len;
    return beam.make_ok_binary(env, name_buf[0..name_len]);
}

const DriverResult = union(enum) {
    ok: *const CudaDriver,
    err: beam.term,
};

fn requireDriver(env: beam.env) DriverResult {
    const d = loadDriver() orelse return .{ .err = beam.make_error_binary(env, "libcuda not found") };
    if (d.init_result != 0) return .{ .err = cuResultError(env, "cuInit", d.init_result) };
    // The current context is thread-local in the CUDA driver; NIF calls can
    // run on any scheduler thread, so (re)set the retained primary context
    // before every driver operation that needs one.
    _ = d.cuInit(0);
    const set_res = d.cuCtxSetCurrent(d.primary_ctx);
    if (set_res != 0) return .{ .err = cuResultError(env, "cuCtxSetCurrent", set_res) };
    return .{ .ok = d };
}

/// Loads PTX assembly text into a CUDA module. The Elixir side extracts the
/// assembly from the `gpu.binary` produced by `GPU.package_binary!/3`.
/// Returns `{:ok, module_handle}`.
pub fn cuda_module_load(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = switch (requireDriver(env)) {
        .ok => |ok_driver| ok_driver,
        .err => |err_term| return err_term,
    };
    const ptx_bin = try beam.get_binary(env, args[0]);

    // cuModuleLoadData expects NUL-terminated data for PTX text.
    const ptx = std.heap.page_allocator.alloc(u8, ptx_bin.size + 1) catch
        return beam.make_error_binary(env, "out of memory");
    defer std.heap.page_allocator.free(ptx);
    @memcpy(ptx[0..ptx_bin.size], ptx_bin.data[0..ptx_bin.size]);
    ptx[ptx_bin.size] = 0;

    var cu_module: ?*anyopaque = null;
    const result = d.cuModuleLoadData(&cu_module, ptx.ptr);
    if (result != 0) return cuResultError(env, "cuModuleLoadData", result);
    return beam.make_ok_term(env, try beam.make(usize, env, @intFromPtr(cu_module.?)));
}

/// Returns `{:ok, function_handle}` for the kernel `name` in `module_handle`.
pub fn cuda_module_get_function(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = switch (requireDriver(env)) {
        .ok => |ok_driver| ok_driver,
        .err => |err_term| return err_term,
    };
    const module_ptr = try beam.get_u64(env, args[0]);
    const name_bin = try beam.get_binary(env, args[1]);

    const name = std.heap.page_allocator.alloc(u8, name_bin.size + 1) catch
        return beam.make_error_binary(env, "out of memory");
    defer std.heap.page_allocator.free(name);
    @memcpy(name[0..name_bin.size], name_bin.data[0..name_bin.size]);
    name[name_bin.size] = 0;

    var function: ?*anyopaque = null;
    const result = d.cuModuleGetFunction(
        &function,
        @ptrFromInt(module_ptr),
        @ptrCast(name.ptr),
    );
    if (result != 0) return cuResultError(env, "cuModuleGetFunction", result);
    return beam.make_ok_term(env, try beam.make(usize, env, @intFromPtr(function.?)));
}

/// Unloads a CUDA module handle.
pub fn cuda_module_unload(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = switch (requireDriver(env)) {
        .ok => |ok_driver| ok_driver,
        .err => |err_term| return err_term,
    };
    const module_ptr = try beam.get_u64(env, args[0]);
    const result = d.cuModuleUnload(@ptrFromInt(module_ptr));
    if (result != 0) return cuResultError(env, "cuModuleUnload", result);
    return beam.make_ok(env);
}

/// Returns `{:ok, device_ptr}` for a device allocation of `size` bytes.
pub fn cuda_mem_alloc(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = switch (requireDriver(env)) {
        .ok => |ok_driver| ok_driver,
        .err => |err_term| return err_term,
    };
    const size = try beam.get_u64(env, args[0]);
    var ptr: usize = 0;
    const result = d.cuMemAlloc(&ptr, size);
    if (result != 0) return cuResultError(env, "cuMemAlloc", result);
    return beam.make_ok_term(env, try beam.make(usize, env, ptr));
}

/// Frees a device allocation.
pub fn cuda_mem_free(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = switch (requireDriver(env)) {
        .ok => |ok_driver| ok_driver,
        .err => |err_term| return err_term,
    };
    const ptr = try beam.get_u64(env, args[0]);
    const result = d.cuMemFree(ptr);
    if (result != 0) return cuResultError(env, "cuMemFree", result);
    return beam.make_ok(env);
}

/// Copies host `data` into device memory at `device_ptr`.
pub fn cuda_memcpy_htod(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = switch (requireDriver(env)) {
        .ok => |ok_driver| ok_driver,
        .err => |err_term| return err_term,
    };
    const dst = try beam.get_u64(env, args[0]);
    const data = try beam.get_binary(env, args[1]);
    const result = d.cuMemcpy(dst, @intFromPtr(data.data), data.size);
    if (result != 0) return cuResultError(env, "cuMemcpy", result);
    return beam.make_ok(env);
}

/// Copies `size` bytes from device memory at `device_ptr` into a host binary.
pub fn cuda_memcpy_dtoh(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = switch (requireDriver(env)) {
        .ok => |ok_driver| ok_driver,
        .err => |err_term| return err_term,
    };
    const src = try beam.get_u64(env, args[0]);
    const size = try beam.get_u64(env, args[1]);

    const buffer = std.heap.page_allocator.alloc(u8, @intCast(size)) catch
        return beam.make_error_binary(env, "out of memory");
    defer std.heap.page_allocator.free(buffer);
    const result = d.cuMemcpy(@intFromPtr(buffer.ptr), src, size);
    if (result != 0) return cuResultError(env, "cuMemcpy", result);
    return beam.make_ok_term(env, beam.make_slice(env, buffer));
}

/// Launches a kernel.
///
/// `args` is a binary of kernel parameters packed in 8-byte slots (the Elixir
/// side encodes `f32` as 4 bytes + 4 padding and `i64`/pointers as 8 bytes),
/// matching the aligned `.param` layout of NVVM kernels.
pub fn cuda_launch_kernel(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
    const d = switch (requireDriver(env)) {
        .ok => |ok_driver| ok_driver,
        .err => |err_term| return err_term,
    };
    const function_ptr = try beam.get_u64(env, args[0]);
    const grid_x = try beam.get_u32(env, args[1]);
    const grid_y = try beam.get_u32(env, args[2]);
    const grid_z = try beam.get_u32(env, args[3]);
    const block_x = try beam.get_u32(env, args[4]);
    const block_y = try beam.get_u32(env, args[5]);
    const block_z = try beam.get_u32(env, args[6]);
    const params_bin = try beam.get_binary(env, args[7]);

    if (params_bin.size == 0 or params_bin.size % 8 != 0)
        return beam.make_error_binary(env, "kernel params must be packed in 8-byte slots");

    const allocator = std.heap.page_allocator;
    const buffer = allocator.alloc(u8, params_bin.size) catch
        return beam.make_error_binary(env, "out of memory");
    defer allocator.free(buffer);
    @memcpy(buffer, params_bin.data[0..params_bin.size]);

    const slot_count = params_bin.size / 8;
    const params = allocator.alloc(?*anyopaque, slot_count) catch
        return beam.make_error_binary(env, "out of memory");
    defer allocator.free(params);
    for (0..slot_count) |i| {
        params[i] = @ptrFromInt(@intFromPtr(buffer.ptr) + i * 8);
    }

    const result = d.cuLaunchKernel(
        @ptrFromInt(function_ptr),
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        0,
        null,
        params.ptr,
        null,
    );
    if (result != 0) return cuResultError(env, "cuLaunchKernel", result);
    return beam.make_ok(env);
}

pub const nifs = .{
    prelude.beaverRawNIF(@This(), "cuda_available", 0),
    prelude.beaverRawNIF(@This(), "cuda_device_count", 0),
    prelude.beaverRawNIF(@This(), "cuda_device_name", 1),
    prelude.beaverRawNIF(@This(), "cuda_module_load", 1),
    prelude.beaverRawNIF(@This(), "cuda_module_get_function", 2),
    prelude.beaverRawNIF(@This(), "cuda_module_unload", 1),
    prelude.beaverRawNIF(@This(), "cuda_mem_alloc", 1),
    prelude.beaverRawNIF(@This(), "cuda_mem_free", 1),
    prelude.beaverRawNIF(@This(), "cuda_memcpy_htod", 2),
    prelude.beaverRawNIF(@This(), "cuda_memcpy_dtoh", 2),
    prelude.beaverRawNIF(@This(), "cuda_launch_kernel", 8),
};

export const cuda_runner_nifs: [nifs.len]e.ErlNifFunc = nifs;
export const cuda_runner_nifs_len: usize = nifs.len;

export fn cuda_runner_open(_: beam.env) void {}
