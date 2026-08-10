const std = @import("std");
const kinda = @import("kinda");
const e = kinda.erl_nif;
const beam = kinda.beam;
const result = kinda.result;
const mlir_capi = @import("mlir_capi.zig");
const diagnostic = @import("diagnostic.zig");
const c = @import("prelude.zig").c;

const Translation = struct {
    const Buffer = std.array_list.Managed(u8);

    buffer: Buffer,

    fn appendString(value: mlir_capi.StringRef.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        self.buffer.appendSlice(value.data[0..value.length]) catch @panic("failed to collect LLVM IR");
    }

    fn translateWithoutDiagnostics(
        environment: beam.env,
        _: c_int,
        args: [*c]const beam.term,
    ) !beam.term {
        const operation = try mlir_capi.Operation.resource.fetch(environment, args[0]);
        var self = @This(){ .buffer = Buffer.init(beam.allocator) };
        defer self.buffer.deinit();

        const status = c.beaverTranslateModuleToLLVMIRText(operation, appendString, &self);
        if (c.beaverLogicalResultIsFailure(status))
            return beam.make_atom(environment, "error");

        return beam.make_slice(environment, self.buffer.items);
    }

    fn translate(
        environment: beam.env,
        argc: c_int,
        args: [*c]const beam.term,
    ) !beam.term {
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

const PTXCompilation = struct {
    const Buffer = std.array_list.Managed(u8);

    output: Buffer,
    error_message: Buffer,

    fn appendOutput(value: mlir_capi.StringRef.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        self.output.appendSlice(value.data[0..value.length]) catch @panic("failed to collect PTX");
    }

    fn appendError(value: mlir_capi.StringRef.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        self.error_message.appendSlice(value.data[0..value.length]) catch @panic("failed to collect LLVM target diagnostic");
    }

    fn compile(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const llvm_ir = try beam.get_binary(environment, args[0]);
        const cpu = try beam.get_binary(environment, args[1]);
        const features = try beam.get_binary(environment, args[2]);
        var self = @This(){
            .output = Buffer.init(beam.allocator),
            .error_message = Buffer.init(beam.allocator),
        };
        defer self.output.deinit();
        defer self.error_message.deinit();

        const status = c.beaverCompileLLVMIRToPTX(
            c.mlirStringRefCreate(llvm_ir.data, llvm_ir.size),
            c.mlirStringRefCreate(cpu.data, cpu.size),
            c.mlirStringRefCreate(features.data, features.size),
            appendOutput,
            appendError,
            &self,
        );
        if (c.beaverLogicalResultIsFailure(status)) {
            const message = if (self.error_message.items.len == 0)
                "LLVM target compilation failed"
            else
                self.error_message.items;
            return beam.make_error_binary(environment, message);
        }

        return beam.make_ok_binary(environment, self.output.items);
    }
};

pub const nifs = .{
    result.nif_with_flags(
        "beaver_raw_translate_module_to_llvm_ir",
        1,
        Translation.translate,
        e.ERL_NIF_DIRTY_JOB_CPU_BOUND,
    ).entry,
    result.nif_with_flags(
        "beaver_raw_compile_llvm_ir_to_ptx",
        3,
        PTXCompilation.compile,
        e.ERL_NIF_DIRTY_JOB_CPU_BOUND,
    ).entry,
};
