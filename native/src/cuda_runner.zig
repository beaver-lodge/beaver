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

/// Encodes an opaque pointer (CUDA module/function/device handles) as a BEAM
/// unsigned 64-bit integer. `beam.make(usize, ...)` is unsuitable here because
/// it routes through `enif_make_int`, which truncates to 32 bits on 64-bit
/// platforms.
fn make_ptr(env: beam.env, ptr: usize) beam.term {
    return e.enif_make_uint64(env, @intCast(ptr));
}
const CudaDriver = struct {
    dynlib: std.DynLib,
    cuInit: *const fn (flags: c_uint) callconv(.c) CudaResult,
    cuDeviceGetCount: *const fn (count: *c_int) callconv(.c) CudaResult,
    cuDeviceGet: *const fn (device: *CuDevice, ordinal: c_int) callconv(.c) CudaResult,
    cuDeviceGetName: *const fn (name: [*c]u8, len: c_int, dev: CuDevice) callconv(.c) CudaResult,
    cuDeviceGetAttribute: *const fn (value: *c_int, attribute: c_int, dev: CuDevice) callconv(.c) CudaResult,
    cuModuleLoadData: *const fn (module: *?*anyopaque, data: *const anyopaque) callconv(.c) CudaResult,
    cuModuleGetFunction: *const fn (function: *?*anyopaque, module: ?*anyopaque, name: [*:0]const u8) callconv(.c) CudaResult,
    cuModuleUnload: *const fn (module: ?*anyopaque) callconv(.c) CudaResult,
    cuFuncSetAttribute: *const fn (function: ?*anyopaque, attribute: c_int, value: c_int) callconv(.c) CudaResult,
    cuMemAlloc: *const fn (ptr: *usize, size: usize) callconv(.c) CudaResult,
    cuMemFree: *const fn (ptr: usize) callconv(.c) CudaResult,
    cuMemcpy: *const fn (dst: usize, src: usize, size: usize) callconv(.c) CudaResult,
    cuDevicePrimaryCtxRetain: *const fn (pctx: *?*anyopaque, dev: CuDevice) callconv(.c) CudaResult,
    cuCtxSetCurrent: *const fn (ctx: ?*anyopaque) callconv(.c) CudaResult,
    cuCtxSynchronize: *const fn () callconv(.c) CudaResult,
    cuLaunchKernel: *const fn (
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
    ) callconv(.c) CudaResult,
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
const cu_device_attribute_compute_capability_major = 75;
const cu_device_attribute_compute_capability_minor = 76;
var driver: ?CudaDriver = null;
var driver_probe_done: bool = false;
var driver_mutex: mutex.Mutex = .{};
fn loadDriver() ?*const CudaDriver {
    driver_mutex.lock();
    defer driver_mutex.unlock();
    if (driver_probe_done) return if (driver) |*d| d else null;
    driver_probe_done = true;
    candidates: for (libcuda_candidates) |path| {
        var lib = std.DynLib.open(path) catch continue;
        var installed = false;
        defer if (!installed) lib.close();
        var candidate: CudaDriver = undefined;
        candidate.dynlib = lib;
        // Function signatures and symbol names have one source of truth: the fields.
        inline for (std.meta.fields(CudaDriver)) |field| {
            if (comptime std.mem.startsWith(u8, field.name, "cu")) {
                const versioned = comptime std.mem.eql(u8, field.name, "cuMemAlloc") or
                    std.mem.eql(u8, field.name, "cuMemFree") or std.mem.eql(u8, field.name, "cuMemcpy");
                // CUDA 12+ may expose incompatible legacy memory entry points.
                @field(candidate, field.name) = (if (versioned) lib.lookup(field.type, field.name ++ "_v2") else null) orelse lib.lookup(field.type, field.name) orelse continue :candidates;
            }
        }
        candidate.init_result = candidate.cuInit(0);
        candidate.primary_ctx = null;
        if (candidate.init_result == 0) {
            var device: CuDevice = undefined;
            _ = candidate.cuDeviceGet(&device, 0);
            _ = candidate.cuDevicePrimaryCtxRetain(&candidate.primary_ctx, device);
            _ = candidate.cuCtxSetCurrent(candidate.primary_ctx);
        }
        driver = candidate;
        installed = true;
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
fn cuda_device_count(d: *const CudaDriver, env: beam.env, _: [*c]const beam.term) !beam.term {
    var count: c_int = 0;
    const result = d.cuDeviceGetCount(&count);
    if (result != 0) return cuResultError(env, "cuDeviceGetCount", result);
    return beam.make_ok_term(env, try beam.make(i32, env, @intCast(count)));
}

/// Returns `{:ok, name}` for the device at `ordinal`, or `{:error, reason}`.
fn cuda_device_name(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
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

/// Returns `{:ok, {major, minor}}` with the compute capability of the device
/// at `ordinal`, or `{:error, reason}`.
fn cuda_device_compute_capability(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const ordinal = try beam.get_i32(env, args[0]);
    var device: CuDevice = undefined;
    const get_result = d.cuDeviceGet(&device, ordinal);
    if (get_result != 0) return cuResultError(env, "cuDeviceGet", get_result);
    var terms: [2]beam.term = undefined;
    inline for (.{ .{ "MAJOR", cu_device_attribute_compute_capability_major }, .{ "MINOR", cu_device_attribute_compute_capability_minor } }, 0..) |attribute, index| {
        var value: c_int = 0;
        const result = d.cuDeviceGetAttribute(&value, attribute[1], device);
        if (result != 0) return cuResultError(env, "cuDeviceGetAttribute(COMPUTE_CAPABILITY_" ++ attribute[0] ++ ")", result);
        terms[index] = try beam.make(i32, env, value);
    }
    return beam.make_ok_term(env, beam.make_tuple(env, &terms));
}

/// Common driver admission; only device-memory/module operations bind a context.
fn cudaNIF(comptime name: []const u8, comptime arity: usize, comptime context: bool) e.ErlNifFunc {
    const operation = @field(@This(), name);
    const Wrapper = struct {
        fn invoke(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const d = loadDriver() orelse return beam.make_error_binary(env, "libcuda not found");
            if (d.init_result != 0) return cuResultError(env, "cuInit", d.init_result);
            if (context) {
                // CUDA context binding is scheduler-thread-local, not process-local.
                _ = d.cuInit(0);
                const result = d.cuCtxSetCurrent(d.primary_ctx);
                if (result != 0) return cuResultError(env, "cuCtxSetCurrent", result);
            }
            return operation(d, env, args);
        }
    };
    return kinda.result.nif("beaver_raw_" ++ name, arity, Wrapper.invoke).entry;
}
fn status(env: beam.env, name: []const u8, result: CudaResult) beam.term {
    return if (result == 0) beam.make_ok(env) else cuResultError(env, name, result);
}

/// Loads PTX assembly text into a CUDA module. The Elixir side extracts the
/// assembly from the `gpu.binary` produced by `GPU.package_binary!/3`.
/// Returns `{:ok, module_handle}`.
fn cuda_module_load(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const ptx_bin = try beam.get_binary(env, args[0]);

    // cuModuleLoadData expects NUL-terminated data for PTX text.
    const ptx = std.heap.page_allocator.dupeZ(u8, ptx_bin.data[0..ptx_bin.size]) catch
        return beam.make_error_binary(env, "out of memory");
    defer std.heap.page_allocator.free(ptx);
    var cu_module: ?*anyopaque = null;
    const result = d.cuModuleLoadData(&cu_module, ptx.ptr);
    if (result != 0) return cuResultError(env, "cuModuleLoadData", result);
    return beam.make_ok_term(env, make_ptr(env, @intFromPtr(cu_module.?)));
}

/// Returns `{:ok, function_handle}` for the kernel `name` in `module_handle`.
fn cuda_module_get_function(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const module_ptr = try beam.get_u64(env, args[0]);
    const name_bin = try beam.get_binary(env, args[1]);
    const name = std.heap.page_allocator.dupeZ(u8, name_bin.data[0..name_bin.size]) catch
        return beam.make_error_binary(env, "out of memory");
    defer std.heap.page_allocator.free(name);
    var function: ?*anyopaque = null;
    const result = d.cuModuleGetFunction(&function, @ptrFromInt(module_ptr), @ptrCast(name.ptr));
    if (result != 0) return cuResultError(env, "cuModuleGetFunction", result);
    return beam.make_ok_term(env, make_ptr(env, @intFromPtr(function.?)));
}

/// Sets a CUDA function attribute, e.g. the dynamic shared memory size in
/// bytes (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8`).
fn cuda_func_set_attribute(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const function_ptr = try beam.get_u64(env, args[0]);
    const attribute = try beam.get_i32(env, args[1]);
    const value = try beam.get_i32(env, args[2]);
    return status(env, "cuFuncSetAttribute", d.cuFuncSetAttribute(@ptrFromInt(function_ptr), attribute, value));
}

/// Unloads a CUDA module handle.
fn cuda_module_unload(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const module_ptr = try beam.get_u64(env, args[0]);
    return status(env, "cuModuleUnload", d.cuModuleUnload(@ptrFromInt(module_ptr)));
}

/// Returns `{:ok, device_ptr}` for a device allocation of `size` bytes.
fn cuda_mem_alloc(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const size = try beam.get_u64(env, args[0]);
    var ptr: usize = 0;
    const result = d.cuMemAlloc(&ptr, size);
    if (result != 0) return cuResultError(env, "cuMemAlloc", result);
    return beam.make_ok_term(env, make_ptr(env, ptr));
}

/// Frees a device allocation.
fn cuda_mem_free(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const ptr = try beam.get_u64(env, args[0]);
    return status(env, "cuMemFree", d.cuMemFree(ptr));
}

/// Blocks until all previously queued work on the current context completes.
fn cuda_synchronize(d: *const CudaDriver, env: beam.env, _: [*c]const beam.term) !beam.term {
    return status(env, "cuCtxSynchronize", d.cuCtxSynchronize());
}

/// Copies host `data` into device memory at `device_ptr`.
fn cuda_memcpy_htod(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const dst = try beam.get_u64(env, args[0]);
    const data = try beam.get_binary(env, args[1]);
    return status(env, "cuMemcpy", d.cuMemcpy(dst, @intFromPtr(data.data), data.size));
}

/// Copies `size` bytes from device memory at `device_ptr` into a host binary.
fn cuda_memcpy_dtoh(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const src = try beam.get_u64(env, args[0]);
    const size = try beam.get_u64(env, args[1]);
    const buffer = std.heap.page_allocator.alloc(u8, @intCast(size)) catch
        return beam.make_error_binary(env, "out of memory");
    defer std.heap.page_allocator.free(buffer);
    const result = d.cuMemcpy(@intFromPtr(buffer.ptr), src, size);
    if (result != 0) return cuResultError(env, "cuMemcpy", result);
    return beam.make_ok_term(env, beam.make_slice(env, buffer));
}

/// Default-stream device copy, with no intermediate host allocation.
fn cuda_memcpy_dtod(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const dst = try beam.get_u64(env, args[0]);
    const src = try beam.get_u64(env, args[1]);
    const size = try beam.get_u64(env, args[2]);
    if (size == 0) return beam.make_ok(env);
    if (dst == 0 or src == 0) return beam.make_error_binary(env, "device copy requires non-null ranges");
    if (dst == src) return beam.make_ok(env);
    if ((if (dst > src) dst - src else src - dst) < size)
        return beam.make_error_binary(env, "overlapping device copy ranges");
    return status(env, "cuMemcpyDtoD", d.cuMemcpy(dst, src, size));
}

/// Launches a kernel.
///
/// `args` is a binary of kernel parameters packed in 8-byte slots (the Elixir
/// side encodes `f32` as 4 bytes + 4 padding and `i64`/pointers as 8 bytes),
/// matching the aligned `.param` layout of NVVM kernels.
fn cuda_launch_kernel(d: *const CudaDriver, env: beam.env, args: [*c]const beam.term) !beam.term {
    const function_ptr = try beam.get_u64(env, args[0]);
    const grid_x = try beam.get_u32(env, args[1]);
    const grid_y = try beam.get_u32(env, args[2]);
    const grid_z = try beam.get_u32(env, args[3]);
    const block_x = try beam.get_u32(env, args[4]);
    const block_y = try beam.get_u32(env, args[5]);
    const block_z = try beam.get_u32(env, args[6]);
    const shared_mem = try beam.get_u32(env, args[7]);
    const params_bin = try beam.get_binary(env, args[8]);
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
    return status(env, "cuLaunchKernel", d.cuLaunchKernel(
        @ptrFromInt(function_ptr),
        grid_x,
        grid_y,
        grid_z,
        block_x,
        block_y,
        block_z,
        shared_mem,
        null,
        params.ptr,
        null,
    ));
}
pub const nifs = .{
    cudaNIF("cuda_memcpy_dtod", 3, true),
    prelude.beaverRawNIF(@This(), "cuda_available", 0),
    cudaNIF("cuda_device_count", 0, false),
    cudaNIF("cuda_device_name", 1, false),
    cudaNIF("cuda_device_compute_capability", 1, false),
    cudaNIF("cuda_module_load", 1, true),
    cudaNIF("cuda_module_get_function", 2, true),
    cudaNIF("cuda_func_set_attribute", 3, true),
    cudaNIF("cuda_module_unload", 1, true),
    cudaNIF("cuda_mem_alloc", 1, true),
    cudaNIF("cuda_mem_free", 1, true),
    cudaNIF("cuda_synchronize", 0, true),
    cudaNIF("cuda_memcpy_htod", 2, true),
    cudaNIF("cuda_memcpy_dtoh", 2, true),
    cudaNIF("cuda_launch_kernel", 9, true),
};
export const cuda_runner_nifs: [nifs.len]e.ErlNifFunc = nifs;
export const cuda_runner_nifs_len: usize = nifs.len;
export fn cuda_runner_open(_: beam.env) void {}
