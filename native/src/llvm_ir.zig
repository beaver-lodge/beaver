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

const PTXCompilation = struct {
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

    pub fn compile_llvm_ir_to_ptx(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
        const llvm_ir = try string_ref.get_binary_as_string_ref(environment, args[0]);
        const cpu = try string_ref.get_binary_as_string_ref(environment, args[1]);
        const features = try string_ref.get_binary_as_string_ref(environment, args[2]);
        var self = @This(){ .output = Buffer.init(beam.allocator), .error_message = Buffer.init(beam.allocator) };
        defer self.output.deinit();
        defer self.error_message.deinit();

        const status = c.beaverCompileLLVMIRToPTX(
            llvm_ir,
            cpu,
            features,
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
    prelude.beaverRawNIFDirtyCPU(Translation, "translate_module_to_llvm_ir", 1),
    prelude.beaverRawNIFDirtyCPU(PTXCompilation, "compile_llvm_ir_to_ptx", 3),
};
