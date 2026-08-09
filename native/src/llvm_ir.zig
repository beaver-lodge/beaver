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
        if (c.mlirLogicalResultIsFailure(status))
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

pub const nifs = .{
    result.nif_with_flags(
        "beaver_raw_translate_module_to_llvm_ir",
        1,
        Translation.translate,
        e.ERL_NIF_DIRTY_JOB_CPU_BOUND,
    ).entry,
};
