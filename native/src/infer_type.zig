const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const m = @import("mlir_capi.zig");
const c = @import("prelude.zig").c;
const callback_args = @import("callback_args.zig");

// Encode while the callback owns its borrowed data; no copied shapes or second
// native result representation are needed after inference returns.
const Collector = struct {
    env: beam.env,
    terms: std.array_list.Managed(beam.term) = .init(beam.allocator),
    failure: ?anyerror = null,
    fn types(count: isize, values: [*c]m.Type.T, data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(data.?));
        for (callback_args.Range(m.Type).slice(count, values)) |value| {
            const term = m.Type.resource.make_kind(self.env, value) catch |err| {
                self.failure = err;
                return;
            };
            self.terms.append(term) catch |err| {
                self.failure = err;
                return;
            };
        }
    }
    fn components(ranked: bool, rank: isize, dims: [*c]const i64, element: m.Type.T, encoding: m.Attribute.T, data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(data.?));
        self.component(ranked, rank, dims, element, encoding) catch |err| {
            self.failure = err;
        };
    }
    fn component(self: *@This(), ranked: bool, rank: isize, dims: [*c]const i64, element: m.Type.T, encoding: m.Attribute.T) !void {
        var shape = beam.make_atom(self.env, "unranked");
        if (ranked) {
            shape = beam.make_term_list(self.env, &.{});
            var remaining: usize = @intCast(rank);
            while (remaining > 0) {
                remaining -= 1;
                shape = kinda.erl_nif.enif_make_list_cell(self.env, beam.make_i64(self.env, dims[remaining]), shape);
            }
        }
        var fields = [_]beam.term{
            shape,
            try callback_args.Optional(m.Type).encode(self.env, element),
            try callback_args.Optional(m.Attribute).encode(self.env, encoding),
        };
        try self.terms.append(beam.make_tuple(self.env, &fields));
    }
};
fn Inference(comptime shaped: bool) type {
    return struct {
        fn collect(env: beam.env, ctx: m.Context.T, args: [*c]const beam.term) !beam.term {
            const location = try m.Location.resource.fetch(env, args[2]);
            const operands = try callback_args.Range(m.Value).decode(env, args[3], beam.allocator);
            defer beam.allocator.free(operands);
            const regions = try callback_args.Range(m.Region).decode(env, args[6], beam.allocator);
            defer beam.allocator.free(regions);
            var collector = Collector{ .env = env };
            defer collector.terms.deinit();
            const infer = if (shaped) c.beaverInferReturnComponents else c.beaverInferReturnTypes;
            const status = infer(
                try @import("string_ref.zig").get_binary_as_string_ref(env, args[0]),
                ctx,
                location,
                @intCast(operands.len),
                operands.ptr,
                try m.Attribute.resource.fetch(env, args[4]),
                if (beam.is_nil2(env, args[5])) .{ .ptr = null } else try m.Attribute.resource.fetch(env, args[5]),
                @intCast(regions.len),
                regions.ptr,
                if (shaped) Collector.components else Collector.types,
                &collector,
            );
            if (collector.failure) |err| return err;
            if (c.beaverLogicalResultIsFailure(status)) {
                c.mlirEmitError(location, if (shaped) "failed to infer shaped operation return components" else "failed to infer operation return types");
                return beam.make_atom(env, "error");
            }
            return beam.make_term_list(env, collector.terms.items);
        }
        fn run(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const ctx = try m.Context.resource.fetch(env, args[1]);
            return @import("diagnostic.zig").call_with_diagnostics(env, ctx, collect, .{ env, ctx, args });
        }
    };
}
pub const nifs = .{
    kinda.result.nif("beaver_raw_infer_return_types", 7, Inference(false).run).entry,
    kinda.result.nif("beaver_raw_infer_return_type_components", 7, Inference(true).run).entry,
};
