const std = @import("std");
const kinda = @import("kinda");
const beam = kinda.beam;
const mlir_capi = @import("mlir_capi.zig");
const diagnostic = @import("diagnostic.zig");
const c = @import("prelude.zig").c;

const Request = struct {
    op_name: mlir_capi.StringRef.T,
    context: mlir_capi.Context.T,
    location: mlir_capi.Location.T,
    operands: []mlir_capi.Value.T,
    attributes: mlir_capi.Attribute.T,
    regions: []mlir_capi.Region.T,

    fn init(environment: beam.env, args: [*c]const beam.term) !@This() {
        if (0 == kinda.erl_nif.enif_is_identical(args[5], beam.make_atom(environment, "nil")))
            return error.UnsupportedProperties;

        const op_name = try beam.get_binary(environment, args[0]);
        const operands = try @import("callback_args.zig").Range(mlir_capi.Value).decode(environment, args[3], beam.allocator);
        errdefer beam.allocator.free(operands);
        const regions = try @import("callback_args.zig").Range(mlir_capi.Region).decode(environment, args[6], beam.allocator);
        errdefer beam.allocator.free(regions);
        return .{
            .op_name = c.mlirStringRefCreate(op_name.data, op_name.size),
            .context = try mlir_capi.Context.resource.fetch(environment, args[1]),
            .location = try mlir_capi.Location.resource.fetch(environment, args[2]),
            .operands = operands,
            .attributes = try mlir_capi.Attribute.resource.fetch(environment, args[4]),
            .regions = regions,
        };
    }

    fn deinit(self: *@This()) void {
        beam.allocator.free(self.operands);
        beam.allocator.free(self.regions);
    }
};

const TypeCollector = struct {
    values: std.array_list.Managed(mlir_capi.Type.T),

    fn init() @This() {
        return .{ .values = std.array_list.Managed(mlir_capi.Type.T).init(beam.allocator) };
    }

    fn deinit(self: *@This()) void {
        self.values.deinit();
    }

    fn append(count: isize, values: [*c]mlir_capi.Type.T, user_data: ?*anyopaque) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        const len: usize = @intCast(count);
        self.values.appendSlice(values[0..len]) catch @panic("failed to collect inferred types");
    }

    fn toTerm(self: *@This(), environment: beam.env) !beam.term {
        return kinda.callback_adapter.handleRange(mlir_capi.Type, environment, self.values.items);
    }
};

const ShapedTypeCollector = struct {
    const Component = struct {
        ranked: bool,
        shape: []i64,
        element_type: mlir_capi.Type.T,
        encoding: mlir_capi.Attribute.T,
    };

    values: std.array_list.Managed(Component),

    fn init() @This() {
        return .{ .values = std.array_list.Managed(Component).init(beam.allocator) };
    }

    fn deinit(self: *@This()) void {
        for (self.values.items) |component| beam.allocator.free(component.shape);
        self.values.deinit();
    }

    fn append(
        ranked: bool,
        rank: isize,
        shape: [*c]const i64,
        element_type: mlir_capi.Type.T,
        encoding: mlir_capi.Attribute.T,
        user_data: ?*anyopaque,
    ) callconv(.c) void {
        const self: *@This() = @ptrCast(@alignCast(user_data orelse return));
        const len: usize = if (ranked) @intCast(rank) else 0;
        const owned_shape = if (len == 0)
            beam.allocator.alloc(i64, 0) catch @panic("failed to allocate inferred shape")
        else
            beam.allocator.dupe(i64, shape[0..len]) catch @panic("failed to copy inferred shape");
        self.values.append(.{
            .ranked = ranked,
            .shape = owned_shape,
            .element_type = element_type,
            .encoding = encoding,
        }) catch @panic("failed to collect inferred shaped type components");
    }

    fn componentTerm(component: Component, environment: beam.env) !beam.term {
        const shape = if (component.ranked) blk: {
            const terms = try beam.allocator.alloc(beam.term, component.shape.len);
            defer beam.allocator.free(terms);
            for (component.shape, terms) |dimension, *term|
                term.* = try beam.make(i64, environment, dimension);
            break :blk beam.make_term_list(environment, terms);
        } else beam.make_atom(environment, "unranked");

        const element_type = if (c.beaverIsNullType(component.element_type))
            beam.make_atom(environment, "nil")
        else
            try mlir_capi.Type.resource.make_kind(environment, component.element_type);

        const encoding = if (c.beaverIsNullAttribute(component.encoding))
            beam.make_atom(environment, "nil")
        else
            try mlir_capi.Attribute.resource.make_kind(environment, component.encoding);

        var fields = [_]beam.term{ shape, element_type, encoding };
        return beam.make_tuple(environment, &fields);
    }

    fn toTerm(self: *@This(), environment: beam.env) !beam.term {
        const terms = try beam.allocator.alloc(beam.term, self.values.items.len);
        defer beam.allocator.free(terms);
        for (self.values.items, terms) |component, *term|
            term.* = try componentTerm(component, environment);
        return beam.make_term_list(environment, terms);
    }
};

fn Inference(comptime shaped: bool) type {
    const Collector = if (shaped) ShapedTypeCollector else TypeCollector;
    const infer = if (shaped) c.mlirInferShapedTypeOpInterfaceInferReturnTypes else c.mlirInferTypeOpInterfaceInferReturnTypes;
    return struct {
        fn collect(environment: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            var request = try Request.init(environment, args);
            defer request.deinit();
            var collector = Collector.init();
            defer collector.deinit();
            const status = infer(
                request.op_name,
                request.context,
                request.location,
                @intCast(request.operands.len),
                request.operands.ptr,
                request.attributes,
                null,
                @intCast(request.regions.len),
                request.regions.ptr,
                Collector.append,
                &collector,
            );
            if (c.beaverLogicalResultIsFailure(status)) {
                c.mlirEmitError(request.location, if (shaped) "failed to infer shaped operation return components" else "failed to infer operation return types");
                return beam.make_atom(environment, "error");
            }
            return collector.toTerm(environment);
        }

        fn run(environment: beam.env, argc: c_int, args: [*c]const beam.term) !beam.term {
            const context = try mlir_capi.Context.resource.fetch(environment, args[1]);
            return diagnostic.call_with_diagnostics(environment, context, collect, .{ environment, argc, args });
        }
    };
}

pub const nifs = .{
    kinda.result.nif("beaver_raw_infer_return_types", 7, Inference(false).run).entry,
    kinda.result.nif("beaver_raw_infer_return_type_components", 7, Inference(true).run).entry,
};
