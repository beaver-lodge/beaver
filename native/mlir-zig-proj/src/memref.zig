const beam = @import("beam");
const std = @import("std");
const mem = @import("std").mem;
const mlir_capi = @import("mlir_capi.zig");
pub const c = @import("prelude.zig");
const e = @import("erl_nif");
const result = kinda.result;
const kinda = @import("kinda");

fn MemRefDescriptorAccessor(comptime MemRefT: type) type {
    return struct {
        fn allocated_ptr(env: beam.env, term: beam.term) !beam.term {
            var descriptor: MemRefT = try beam.fetch_resource(MemRefT, env, MemRefT.resource_type, term);
            return try mlir_capi.OpaquePtr.resource.make(env, @ptrCast(descriptor.allocated));
        }
        fn aligned_ptr(env: beam.env, term: beam.term) !beam.term {
            var descriptor: MemRefT = try beam.fetch_resource(MemRefT, env, MemRefT.resource_type, term);
            return try mlir_capi.OpaquePtr.resource.make(env, @ptrCast(descriptor.aligned));
        }
        fn fetch_ptr_or_nil(env: beam.env, ptr_term: beam.term) !MemRefT.ElementResourceKind.Ptr.T {
            if (MemRefT.ElementResourceKind.Ptr.resource.fetch(env, ptr_term)) |value| {
                return value;
            } else |err| {
                if (try beam.is_nil(env, ptr_term)) {
                    return null;
                } else {
                    return err;
                }
            }
        }
        fn offset(env: beam.env, term: beam.term) !beam.term {
            var descriptor = try beam.fetch_resource(MemRefT, env, MemRefT.resource_type, term);
            return beam.make_i64(env, descriptor.offset);
        }
    };
}

fn memref_module_name(comptime resource_kind: type, comptime rank: i32) []const u8 {
    return resource_kind.module_name ++ ".MemRef." ++ @tagName(@as(MemRefRankType, @enumFromInt(rank)));
}

fn UnrankMemRefDescriptor(comptime ResourceKind: type) type {
    return extern struct {
        pub fn make(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            var allocated: ResourceKind.Ptr.T = try MemRefDescriptorAccessor(@This()).fetch_ptr_or_nil(env, args[0]);
            var aligned: ResourceKind.Ptr.T = try MemRefDescriptorAccessor(@This()).fetch_ptr_or_nil(env, args[1]);
            var offset: mlir_capi.I64.T = try mlir_capi.I64.resource.fetch(env, args[2]);
            const kind: type = dataKindToMemrefKind(ResourceKind);
            var descriptor: UnrankMemRefDescriptor(ResourceKind) = undefined;
            if (allocated == null) {
                descriptor = .{
                    .offset = offset,
                };
            } else {
                descriptor = .{
                    .allocated = allocated,
                    .aligned = aligned,
                    .offset = offset,
                };
            }
            return try kind.per_rank_resource_kinds[0].resource.make(env, descriptor);
        }
        pub const maker = .{ make, 5 };
        pub const module_name = memref_module_name(ResourceKind, 0);
        const ElementResourceKind = ResourceKind;
        const T = ResourceKind.T;
        allocated: ?*T = null,
        aligned: ?*T = null,
        offset: i64 = undefined,
        pub var resource_type: beam.resource_type = undefined;
        fn allocated_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            return try MemRefDescriptorAccessor(@This()).allocated_ptr(env, args[0]);
        }
        fn aligned_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            return try MemRefDescriptorAccessor(@This()).aligned_ptr(env, args[0]);
        }
        fn get_offset(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            return try MemRefDescriptorAccessor(@This()).offset(env, args[0]);
        }
        pub const nifs = .{ result.nif(module_name ++ ".allocated", 1, allocated_ptr).entry, result.nif(module_name ++ ".aligned", 1, aligned_ptr).entry, result.nif(module_name ++ ".offset", 1, get_offset).entry };
    };
}

fn MemRefDescriptor(comptime ResourceKind: type, comptime N: usize) type {
    return extern struct {
        const T = ResourceKind.T;
        allocated: ?*T = null,
        aligned: ?*T = null,
        offset: i64 = undefined,
        sizes: [N]i64 = undefined,
        strides: [N]i64 = undefined,
        fn populate(self: *@This(), allocated: *T, aligned: *T, offset: i64, sizes: []i64, strides: []i64) void {
            self.allocated = allocated;
            self.aligned = aligned;
            self.offset = offset;
            const rank = sizes.len;
            mem.copy(i64, self.sizes[0..rank], sizes[0..rank]);
            mem.copy(i64, self.strides[0..rank], strides[0..rank]);
        }
        fn populate2(self: *@This(), offset: i64, sizes: []i64, strides: []i64) void {
            self.offset = offset;
            const rank = sizes.len;
            mem.copy(i64, self.sizes[0..rank], sizes[0..rank]);
            mem.copy(i64, self.strides[0..rank], strides[0..rank]);
        }
        pub fn make(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            const Error = error{
                SizesAndStridesOfDifferentLength,
                WrongSizesForRank,
                WrongStridesForRank,
            };
            var allocated: ResourceKind.Ptr.T = try MemRefDescriptorAccessor(@This()).fetch_ptr_or_nil(env, args[0]);
            var aligned: ResourceKind.Ptr.T = try MemRefDescriptorAccessor(@This()).fetch_ptr_or_nil(env, args[1]);
            var offset: mlir_capi.I64.T = try mlir_capi.I64.resource.fetch(env, args[2]);
            const sizes = try beam.get_slice_of(i64, env, args[3]);
            defer beam.allocator.free(sizes);
            const strides = try beam.get_slice_of(i64, env, args[4]);
            defer beam.allocator.free(strides);
            if (sizes.len != strides.len) {
                return Error.SizesAndStridesOfDifferentLength;
            }
            const kind: type = dataKindToMemrefKind(ResourceKind);
            comptime var rank = N;
            if (rank != sizes.len) {
                return Error.WrongSizesForRank;
            }
            if (rank != strides.len) {
                return Error.WrongStridesForRank;
            }
            var descriptor: MemRefDescriptor(ResourceKind, rank) = .{};
            if (allocated == null) {
                descriptor.populate2(offset, sizes, strides);
            } else {
                descriptor.populate(allocated, aligned, offset, sizes, strides);
            }
            return try kind.per_rank_resource_kinds[rank].resource.make(env, descriptor);
        }
        pub const maker = .{ make, 5 };
        pub const ElementResourceKind = ResourceKind;
        pub const module_name = memref_module_name(ResourceKind, N);
        pub var resource_type: beam.resource_type = undefined;
        fn allocated_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            return try MemRefDescriptorAccessor(@This()).allocated_ptr(env, args[0]);
        }
        fn aligned_ptr(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            return try MemRefDescriptorAccessor(@This()).aligned_ptr(env, args[0]);
        }
        fn get_offset(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            return try MemRefDescriptorAccessor(@This()).offset(env, args[0]);
        }
        fn get_sizes(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            comptime var rank = N;
            var descriptor: @This() = try beam.fetch_resource(@This(), env, @This().resource_type, args[0]);
            var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(rank));
            defer beam.allocator.free(ret);
            var i: usize = 0;
            while (i < rank) : ({
                i += 1;
            }) {
                ret[@intCast(i)] = beam.make_i64(env, descriptor.sizes[i]);
            }
            return beam.make_term_list(env, ret);
        }
        fn get_strides(env: beam.env, _: c_int, args: [*c]const beam.term) !beam.term {
            comptime var rank = N;
            var descriptor: @This() = try beam.fetch_resource(@This(), env, @This().resource_type, args[0]);
            var ret: []beam.term = try beam.allocator.alloc(beam.term, @intCast(rank));
            defer beam.allocator.free(ret);
            var i: usize = 0;
            while (i < rank) : ({
                i += 1;
            }) {
                ret[@intCast(i)] = beam.make_i64(env, descriptor.strides[i]);
            }
            return beam.make_term_list(env, ret);
        }
        pub const nifs = .{
            result.nif(module_name ++ ".allocated", 1, allocated_ptr).entry,
            result.nif(module_name ++ ".aligned", 1, aligned_ptr).entry,
            result.nif(module_name ++ ".offset", 1, get_offset).entry,
            result.nif(module_name ++ ".sizes", 1, get_sizes).entry,
            result.nif(module_name ++ ".strides", 1, get_strides).entry,
        };
    };
}

const Complex = struct {
    fn of(comptime ElementKind: type) type {
        return struct {
            const T = extern struct {
                i: ElementKind.T,
                r: ElementKind.T,
            };
        };
    }
    const F32 = kinda.ResourceKind(Complex.of(mlir_capi.F32).T, "Elixir.Beaver.Native.Complex.F32");
};

const MemRefDataType = enum {
    @"Complex.F32",
    U8,
    U16,
    U32,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
};

fn dataTypeToResourceKind(comptime self: MemRefDataType) type {
    return switch (self) {
        .@"Complex.F32" => Complex.F32,
        .U8 => mlir_capi.U8,
        .U16 => mlir_capi.U16,
        .U32 => mlir_capi.U32,
        .F32 => mlir_capi.F32,
        .F64 => mlir_capi.F64,
        .I8 => mlir_capi.I8,
        .I16 => mlir_capi.I16,
        .I32 => mlir_capi.I32,
        .I64 => mlir_capi.I64,
    };
}

fn dataKindToDataType(comptime self: type) MemRefDataType {
    return switch (self) {
        Complex.F32 => MemRefDataType.@"Complex.F32",
        mlir_capi.U8 => MemRefDataType.U8,
        mlir_capi.U16 => MemRefDataType.U16,
        mlir_capi.U32 => MemRefDataType.U32,
        mlir_capi.F32 => MemRefDataType.F32,
        mlir_capi.F64 => MemRefDataType.F64,
        mlir_capi.I8 => MemRefDataType.I8,
        mlir_capi.I16 => MemRefDataType.I16,
        mlir_capi.I32 => MemRefDataType.I32,
        mlir_capi.I64 => MemRefDataType.I64,
        else => unreachable(),
    };
}

const memref_kinds = .{
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.@"Complex.F32")),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.U8)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.U16)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.U32)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.F32)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.F64)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.I8)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.I16)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.I32)),
    BeaverMemRef(dataTypeToResourceKind(MemRefDataType.I64)),
};

fn dataKindToMemrefKind(comptime self: type) type {
    const dt = dataKindToDataType(self);
    const index = @intFromEnum(dt);
    return memref_kinds[index];
}

const MemRefRankType = enum {
    DescriptorUnranked,
    Descriptor1D,
    Descriptor2D,
    Descriptor3D,
    Descriptor4D,
    Descriptor5D,
    Descriptor6D,
    Descriptor7D,
    Descriptor8D,
    Descriptor9D,
};

fn BeaverMemRef(comptime ResourceKind: type) type {
    return struct {
        const data_kind_name = ResourceKind.module_name;
        pub const nifs =
            per_rank_resource_kinds[0].nifs ++
            per_rank_resource_kinds[1].nifs ++
            per_rank_resource_kinds[2].nifs ++
            per_rank_resource_kinds[3].nifs ++
            per_rank_resource_kinds[4].nifs ++
            per_rank_resource_kinds[5].nifs;
        fn MemRefOfRank(comptime rank: u8) type {
            if (rank == 0) {
                return kinda.ResourceKind2(UnrankMemRefDescriptor(ResourceKind));
            } else {
                return kinda.ResourceKind2(MemRefDescriptor(ResourceKind, rank));
            }
        }
        const per_rank_resource_kinds = .{
            MemRefOfRank(0),
            MemRefOfRank(1),
            MemRefOfRank(2),
            MemRefOfRank(3),
            MemRefOfRank(4),
            MemRefOfRank(5),
        };
        fn open(env: beam.env) void {
            comptime var i = 0;
            inline while (i < per_rank_resource_kinds.len) : (i += 1) {
                per_rank_resource_kinds[i].open_all(env);
            }
        }
    };
}

pub const nifs = Complex.F32.nifs ++
    dataKindToMemrefKind(Complex.F32).nifs ++
    dataKindToMemrefKind(mlir_capi.U8).nifs ++
    dataKindToMemrefKind(mlir_capi.U16).nifs ++
    dataKindToMemrefKind(mlir_capi.U32).nifs ++
    dataKindToMemrefKind(mlir_capi.F32).nifs ++
    dataKindToMemrefKind(mlir_capi.F64).nifs ++
    dataKindToMemrefKind(mlir_capi.I8).nifs ++
    dataKindToMemrefKind(mlir_capi.I16).nifs ++
    dataKindToMemrefKind(mlir_capi.I32).nifs ++
    dataKindToMemrefKind(mlir_capi.I64).nifs;

pub fn open_all(env: beam.env) void {
    comptime var i = 0;
    inline while (i < memref_kinds.len) : (i += 1) {
        memref_kinds[i].open(env);
    }
    Complex.F32.open_all(env);
}
