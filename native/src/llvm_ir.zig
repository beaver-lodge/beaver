const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const mlir_capi = @import("mlir_capi.zig");
const diagnostic = @import("diagnostic.zig");
const prelude = @import("prelude.zig");
const c = prelude.c;
const string_ref = @import("string_ref.zig");
const Buffer = string_ref.TextBuffer;

const Translation = struct {
    fn translateWithoutDiagnostics(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const operation = try mlir_capi.Operation.resource.fetch(environment, args[0]);
        var buffer = Buffer.init(beam.allocator);
        defer buffer.deinit();

        const status = c.beaverTranslateModuleToLLVMIRText(operation, string_ref.collectStringRef, &buffer);
        if (c.beaverLogicalResultIsFailure(status))
            return beam.make_atom(environment, "error");

        return beam.make_slice(environment, buffer.items);
    }

    pub fn translate_module_to_llvm_ir(environment: beam.env, argc: c_int, args: [*c]const beam.term) !beam.term {
        const operation = try mlir_capi.Operation.resource.fetch(environment, args[0]);
        const context = c.mlirOperationGetContext(operation);
        return diagnostic.call_with_diagnostics(
            environment,
            context,
            translateWithoutDiagnostics,
            .{ environment, argc, args },
        );
    }
};

const Compilation = struct {
    output: Buffer,
    error_message: Buffer,

    fn appendOutput(value: mlir_capi.StringRef.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        string_ref.collectStringRef(value, &self.output);
    }

    fn appendError(value: mlir_capi.StringRef.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        string_ref.collectStringRef(value, &self.error_message);
    }

    fn init() @This() {
        return .{ .output = Buffer.init(beam.allocator), .error_message = Buffer.init(beam.allocator) };
    }
    fn deinit(self: *@This()) void {
        self.output.deinit();
        self.error_message.deinit();
    }
    fn failure(self: *@This(), env: beam.env) beam.term {
        const message = if (self.error_message.items.len == 0) "LLVM target compilation failed" else self.error_message.items;
        return beam.make_error_binary(env, message);
    }

    pub fn compile_llvm_ir_to_ptx(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const llvm_ir = try string_ref.get_binary_as_string_ref(environment, args[0]);
        const cpu = try string_ref.get_binary_as_string_ref(environment, args[1]);
        const features = try string_ref.get_binary_as_string_ref(environment, args[2]);
        var self = Compilation.init();
        defer self.deinit();

        const status = c.beaverCompileLLVMIRToPTX(
            llvm_ir,
            cpu,
            features,
            appendOutput,
            appendError,
            &self,
        );
        if (c.beaverLogicalResultIsFailure(status)) return self.failure(environment);

        return beam.make_ok_binary(environment, self.output.items);
    }
};

const HostTarget = struct {
    env: beam.env,
    parts: [3]beam.term = undefined,
    index: usize = 0,

    fn collect(value: mlir_capi.StringRef.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        if (self.index >= 3) return;
        const slice = if (value.data != null and value.length > 0) value.data[0..value.length] else "";
        self.parts[self.index] = beam.make_slice(self.env, slice);
        self.index += 1;
    }

    pub fn llvm_host_target(environment: beam.env, _: c_int, _: [*c]const beam.term) !beam.term {
        var state = @This(){ .env = environment };
        c.beaverLLVMGetHostTarget(collect, &state);
        while (state.index < 3) : (state.index += 1) {
            state.parts[state.index] = beam.make_slice(environment, "");
        }
        return beam.make_tuple(environment, &state.parts);
    }
};

const ObjectCompilation = struct {
    pub fn compile_to_object(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const module_op = if (beam.is_nil2(environment, args[0]))
            mlir_capi.Operation.T{ .ptr = null }
        else
            try mlir_capi.Operation.resource.fetch(environment, args[0]);
        const llvm_ir = if (beam.is_nil2(environment, args[1]))
            mlir_capi.StringRef.T{ .data = null, .length = 0 }
        else
            try string_ref.get_binary_as_string_ref(environment, args[1]);
        const triple = try string_ref.get_binary_as_string_ref(environment, args[2]);
        const cpu = try string_ref.get_binary_as_string_ref(environment, args[3]);
        const features = try string_ref.get_binary_as_string_ref(environment, args[4]);
        const opt_level = try beam.get(c_uint, environment, args[5]);
        const flags = try beam.get(c_uint, environment, args[6]);
        const pic = (flags & 1) != 0;
        const emit_asm = (flags & 2) != 0;

        var self = Compilation.init();
        defer self.deinit();

        var timings = [_]u64{ 0, 0, 0 };
        const status = c.beaverCompileToObject(
            module_op,
            llvm_ir,
            triple,
            cpu,
            features,
            opt_level,
            pic,
            emit_asm,
            &timings,
            Compilation.appendOutput,
            Compilation.appendError,
            &self,
        );
        if (c.beaverLogicalResultIsFailure(status)) return self.failure(environment);

        var timing_terms = [_]beam.term{
            beam.make_u64(environment, timings[0]),
            beam.make_u64(environment, timings[1]),
            beam.make_u64(environment, timings[2]),
        };
        var ok_terms = [_]beam.term{
            beam.make_atom(environment, "ok"),
            beam.make_slice(environment, self.output.items),
            beam.make_tuple(environment, &timing_terms),
        };
        return beam.make_tuple(environment, &ok_terms);
    }

    pub fn extract_function_module(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const module_op = try mlir_capi.Operation.resource.fetch(environment, args[0]);
        const func_name = try string_ref.get_binary_as_string_ref(environment, args[1]);
        const keep_globals = try beam.get_bool(environment, args[2]);
        const extracted = c.beaverExtractFunctionModule(module_op, func_name, keep_globals);
        if (extracted.ptr == null) return beam.make_nil(environment);
        return mlir_capi.Module.resource.make_kind(environment, extracted);
    }
};

pub const nifs = .{
    prelude.beaverRawNIFDirtyCPU(Translation, "translate_module_to_llvm_ir", 1),
    prelude.beaverRawNIFDirtyCPU(Compilation, "compile_llvm_ir_to_ptx", 3),
    prelude.beaverRawNIF(HostTarget, "llvm_host_target", 0),
    prelude.beaverRawNIFDirtyCPU(ObjectCompilation, "compile_to_object", 7),
    prelude.beaverRawNIFDirtyCPU(ObjectCompilation, "extract_function_module", 3),
};
