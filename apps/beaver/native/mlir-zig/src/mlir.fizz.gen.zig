const c = @cImport({
  @cDefine("_NO_CRT_STDIO_INLINE", "1");
  @cInclude("/Users/tsai/oss/beaver/apps/beaver/native/wrapper.h");
});
const beam = @import("beam.zig");
const e = @import("erl_nif.zig");
pub var resource_type_void_ptr: beam.resource_type = undefined;
pub var resource_type_const_void_ptr: beam.resource_type = undefined;
pub var resource_type__nullable_fn__nullable__pointer_anyopaque__callconv__C__void: beam.resource_type = undefined;
pub var resource_type__nullable_fn__nullable__pointer_anyopaque___isize___c_struct_MlirAffineMap__callconv__C__void: beam.resource_type = undefined;
pub var resource_type__nullable_fn_c_struct_MlirDiagnostic____nullable__pointer_anyopaque__callconv__C__c_struct_MlirLogicalResult: beam.resource_type = undefined;
pub var resource_type__nullable_fn_c_struct_MlirOperation___bool____nullable__pointer_anyopaque__callconv__C__void: beam.resource_type = undefined;
pub var resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void: beam.resource_type = undefined;
pub var resource_type__nullable_fn_isize____c_ptr_c_struct_MlirType____nullable__pointer_anyopaque__callconv__C__void: beam.resource_type = undefined;
pub var resource_type__c_ptr__nullable__pointer_anyopaque: beam.resource_type = undefined;
pub var resource_type__c_ptr_c_struct_MlirAffineExpr: beam.resource_type = undefined;
pub var resource_type__c_ptr_c_struct_MlirAffineMap: beam.resource_type = undefined;
pub var resource_type__c_ptr_c_struct_MlirDialectHandle: beam.resource_type = undefined;
pub var resource_type__c_ptr_c_struct_MlirOperationState: beam.resource_type = undefined;
pub var resource_type__c_ptr_c_struct_MlirRegion: beam.resource_type = undefined;
pub var resource_type__c_ptr_c_struct_MlirStringRef: beam.resource_type = undefined;
pub var resource_type__c_ptr_c_struct_MlirValue: beam.resource_type = undefined;
pub var resource_type__c_ptr_c_uint: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_bool: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirAffineExpr: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirAttribute: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirBlock: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirLocation: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirNamedAttribute: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirRegion: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirStringRef: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirType: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_struct_MlirValue: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_int: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_c_uint: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_f32: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_f64: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_i16: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_i32: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_i64: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_i8: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_u16: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_u32: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_u64: beam.resource_type = undefined;
pub var resource_type__c_ptr_const_u8: beam.resource_type = undefined;
pub var resource_type__c_ptr_f64: beam.resource_type = undefined;
pub var resource_type__c_ptr_i64: beam.resource_type = undefined;
pub var resource_type__c_ptr_isize: beam.resource_type = undefined;
pub var resource_type__c_ptr_u64: beam.resource_type = undefined;
pub var resource_type_bool: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirAffineExpr: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirAffineMap: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirAttribute: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirBlock: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirContext: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirDiagnostic: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirDialect: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirDialectHandle: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirDialectRegistry: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirExecutionEngine: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirExternalPass: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirExternalPassCallbacks: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirIdentifier: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirIntegerSet: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirLocation: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirLogicalResult: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirModule: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirNamedAttribute: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirOpPassManager: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirOpPrintingFlags: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirOperation: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirOperationState: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirPass: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirPassManager: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirRegion: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirStringRef: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirSymbolTable: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirType: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirTypeID: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirTypeIDAllocator: beam.resource_type = undefined;
pub var resource_type_c_struct_MlirValue: beam.resource_type = undefined;
pub var resource_type_c_int: beam.resource_type = undefined;
pub var resource_type_c_uint: beam.resource_type = undefined;
pub var resource_type_f32: beam.resource_type = undefined;
pub var resource_type_f64: beam.resource_type = undefined;
pub var resource_type_i16: beam.resource_type = undefined;
pub var resource_type_i32: beam.resource_type = undefined;
pub var resource_type_i64: beam.resource_type = undefined;
pub var resource_type_i8: beam.resource_type = undefined;
pub var resource_type_isize: beam.resource_type = undefined;
pub var resource_type_u16: beam.resource_type = undefined;
pub var resource_type_u32: beam.resource_type = undefined;
pub var resource_type_u64: beam.resource_type = undefined;
pub var resource_type_u8: beam.resource_type = undefined;
pub var resource_type_usize: beam.resource_type = undefined;
pub var resource_type_void: beam.resource_type = undefined;

fn mlirRegisterTransformsViewOpGraphWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsViewOpGraph();
}

export fn fizz_nif_mlirRegisterTransformsViewOpGraph(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsViewOpGraphWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsViewOpGraphWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsViewOpGraph();
}

export fn fizz_nif_mlirCreateTransformsViewOpGraph(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsViewOpGraphWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsTopologicalSortWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsTopologicalSort();
}

export fn fizz_nif_mlirRegisterTransformsTopologicalSort(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsTopologicalSortWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsTopologicalSortWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsTopologicalSort();
}

export fn fizz_nif_mlirCreateTransformsTopologicalSort(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsTopologicalSortWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsSymbolPrivatizeWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsSymbolPrivatize();
}

export fn fizz_nif_mlirRegisterTransformsSymbolPrivatize(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsSymbolPrivatizeWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsSymbolPrivatizeWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsSymbolPrivatize();
}

export fn fizz_nif_mlirCreateTransformsSymbolPrivatize(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsSymbolPrivatizeWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsSymbolDCEWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsSymbolDCE();
}

export fn fizz_nif_mlirRegisterTransformsSymbolDCE(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsSymbolDCEWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsSymbolDCEWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsSymbolDCE();
}

export fn fizz_nif_mlirCreateTransformsSymbolDCE(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsSymbolDCEWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsStripDebugInfoWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsStripDebugInfo();
}

export fn fizz_nif_mlirRegisterTransformsStripDebugInfo(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsStripDebugInfoWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsStripDebugInfoWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsStripDebugInfo();
}

export fn fizz_nif_mlirCreateTransformsStripDebugInfo(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsStripDebugInfoWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsSCCPWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsSCCP();
}

export fn fizz_nif_mlirRegisterTransformsSCCP(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsSCCPWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsSCCPWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsSCCP();
}

export fn fizz_nif_mlirCreateTransformsSCCP(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsSCCPWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsPrintOpStatsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsPrintOpStats();
}

export fn fizz_nif_mlirRegisterTransformsPrintOpStats(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsPrintOpStatsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsPrintOpStatsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsPrintOpStats();
}

export fn fizz_nif_mlirCreateTransformsPrintOpStats(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsPrintOpStatsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsLoopInvariantCodeMotionWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsLoopInvariantCodeMotion();
}

export fn fizz_nif_mlirRegisterTransformsLoopInvariantCodeMotion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsLoopInvariantCodeMotionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsLoopInvariantCodeMotionWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsLoopInvariantCodeMotion();
}

export fn fizz_nif_mlirCreateTransformsLoopInvariantCodeMotion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsLoopInvariantCodeMotionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsLocationSnapshotWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsLocationSnapshot();
}

export fn fizz_nif_mlirRegisterTransformsLocationSnapshot(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsLocationSnapshotWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsLocationSnapshotWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsLocationSnapshot();
}

export fn fizz_nif_mlirCreateTransformsLocationSnapshot(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsLocationSnapshotWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsInlinerWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsInliner();
}

export fn fizz_nif_mlirRegisterTransformsInliner(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsInlinerWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsInlinerWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsInliner();
}

export fn fizz_nif_mlirCreateTransformsInliner(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsInlinerWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsControlFlowSinkWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsControlFlowSink();
}

export fn fizz_nif_mlirRegisterTransformsControlFlowSink(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsControlFlowSinkWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsControlFlowSinkWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsControlFlowSink();
}

export fn fizz_nif_mlirCreateTransformsControlFlowSink(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsControlFlowSinkWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsCanonicalizerWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsCanonicalizer();
}

export fn fizz_nif_mlirRegisterTransformsCanonicalizer(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsCanonicalizerWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsCanonicalizerWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsCanonicalizer();
}

export fn fizz_nif_mlirCreateTransformsCanonicalizer(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsCanonicalizerWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsCSEWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsCSE();
}

export fn fizz_nif_mlirRegisterTransformsCSE(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsCSEWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateTransformsCSEWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateTransformsCSE();
}

export fn fizz_nif_mlirCreateTransformsCSE(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateTransformsCSEWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterTransformsPassesWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterTransformsPasses();
}

export fn fizz_nif_mlirRegisterTransformsPasses(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterTransformsPassesWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirInferTypeOpInterfaceInferReturnTypesWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype, arg5: anytype, arg6: anytype, arg7: anytype, arg8: anytype, arg9: anytype) void {
  ret.* = c.mlirInferTypeOpInterfaceInferReturnTypes(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
}

export fn fizz_nif_mlirInferTypeOpInterfaceInferReturnTypes(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirStringRef = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirStringRef, args[0]);
  var arg1: c.struct_MlirContext = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirContext, args[1]);
  var arg2: c.struct_MlirLocation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirLocation, args[2]);
  var arg3: isize = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_isize, args[3]);
  var arg4: [*c]c.struct_MlirValue = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type__c_ptr_c_struct_MlirValue, args[4]);
  var arg5: c.struct_MlirAttribute = undefined; arg5 = beam.fetch_resource(arg5, env, resource_type_c_struct_MlirAttribute, args[5]);
  var arg6: isize = undefined; arg6 = beam.fetch_resource(arg6, env, resource_type_isize, args[6]);
  var arg7: [*c]c.struct_MlirRegion = undefined; arg7 = beam.fetch_resource(arg7, env, resource_type__c_ptr_c_struct_MlirRegion, args[7]);
  var arg8: ?fn(isize, [*c]c.struct_MlirType, ?*anyopaque) callconv(.C) void = undefined; arg8 = beam.fetch_resource(arg8, env, resource_type__nullable_fn_isize____c_ptr_c_struct_MlirType____nullable__pointer_anyopaque__callconv__C__void, args[8]);
  var arg9: ?*anyopaque = undefined; arg9 = beam.fetch_resource(arg9, env, resource_type_void_ptr, args[9]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLogicalResult, @sizeOf(c.struct_MlirLogicalResult));

  const RType = c.struct_MlirLogicalResult;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirInferTypeOpInterfaceInferReturnTypesWrapper(obj, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
  return e.enif_make_resource(env, ptr);
}

fn mlirInferTypeOpInterfaceTypeIDWrapper(ret: anytype, ) void {
  ret.* = c.mlirInferTypeOpInterfaceTypeID();
}

export fn fizz_nif_mlirInferTypeOpInterfaceTypeID(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirTypeID, @sizeOf(c.struct_MlirTypeID));

  const RType = c.struct_MlirTypeID;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirInferTypeOpInterfaceTypeIDWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationImplementsInterfaceStaticWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationImplementsInterfaceStatic(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationImplementsInterfaceStatic(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirStringRef = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirStringRef, args[0]);
  var arg1: c.struct_MlirContext = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirContext, args[1]);
  var arg2: c.struct_MlirTypeID = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirTypeID, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationImplementsInterfaceStaticWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationImplementsInterfaceWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationImplementsInterface(arg0, arg1);
}

export fn fizz_nif_mlirOperationImplementsInterface(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: c.struct_MlirTypeID = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirTypeID, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationImplementsInterfaceWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetIsConstraintEqWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIntegerSetIsConstraintEq(arg0, arg1);
}

export fn fizz_nif_mlirIntegerSetIsConstraintEq(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetIsConstraintEqWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetConstraintWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIntegerSetGetConstraint(arg0, arg1);
}

export fn fizz_nif_mlirIntegerSetGetConstraint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetConstraintWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetNumInequalitiesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetGetNumInequalities(arg0);
}

export fn fizz_nif_mlirIntegerSetGetNumInequalities(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetNumInequalitiesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetNumEqualitiesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetGetNumEqualities(arg0);
}

export fn fizz_nif_mlirIntegerSetGetNumEqualities(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetNumEqualitiesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetNumConstraintsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetGetNumConstraints(arg0);
}

export fn fizz_nif_mlirIntegerSetGetNumConstraints(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetNumConstraintsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetNumInputsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetGetNumInputs(arg0);
}

export fn fizz_nif_mlirIntegerSetGetNumInputs(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetNumInputsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetNumSymbolsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetGetNumSymbols(arg0);
}

export fn fizz_nif_mlirIntegerSetGetNumSymbols(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetNumSymbolsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetNumDimsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetGetNumDims(arg0);
}

export fn fizz_nif_mlirIntegerSetGetNumDims(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetNumDimsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetIsCanonicalEmptyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetIsCanonicalEmpty(arg0);
}

export fn fizz_nif_mlirIntegerSetIsCanonicalEmpty(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetIsCanonicalEmptyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetReplaceGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirIntegerSetReplaceGet(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirIntegerSetReplaceGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);
  var arg1: [*c]const c.struct_MlirAffineExpr = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__c_ptr_const_c_struct_MlirAffineExpr, args[1]);
  var arg2: [*c]const c.struct_MlirAffineExpr = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirAffineExpr, args[2]);
  var arg3: isize = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_isize, args[3]);
  var arg4: isize = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_isize, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirIntegerSet, @sizeOf(c.struct_MlirIntegerSet));

  const RType = c.struct_MlirIntegerSet;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetReplaceGetWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype, arg5: anytype) void {
  ret.* = c.mlirIntegerSetGet(arg0, arg1, arg2, arg3, arg4, arg5);
}

export fn fizz_nif_mlirIntegerSetGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);
  var arg3: isize = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_isize, args[3]);
  var arg4: [*c]const c.struct_MlirAffineExpr = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type__c_ptr_const_c_struct_MlirAffineExpr, args[4]);
  var arg5: [*c]const bool = undefined; arg5 = beam.fetch_resource(arg5, env, resource_type__c_ptr_const_bool, args[5]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirIntegerSet, @sizeOf(c.struct_MlirIntegerSet));

  const RType = c.struct_MlirIntegerSet;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetWrapper(obj, arg0, arg1, arg2, arg3, arg4, arg5);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetEmptyGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirIntegerSetEmptyGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirIntegerSetEmptyGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirIntegerSet, @sizeOf(c.struct_MlirIntegerSet));

  const RType = c.struct_MlirIntegerSet;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetEmptyGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetDumpWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetDump(arg0);
}

export fn fizz_nif_mlirIntegerSetDump(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetDumpWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetPrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirIntegerSetPrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirIntegerSetPrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetPrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIntegerSetEqual(arg0, arg1);
}

export fn fizz_nif_mlirIntegerSetEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);
  var arg1: c.struct_MlirIntegerSet = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirIntegerSet, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerSetGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerSetGetContext(arg0);
}

export fn fizz_nif_mlirIntegerSetGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIntegerSet = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIntegerSet, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerSetGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirExecutionEngineDumpToObjectFileWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirExecutionEngineDumpToObjectFile(arg0, arg1);
}

export fn fizz_nif_mlirExecutionEngineDumpToObjectFile(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirExecutionEngine = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirExecutionEngine, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirExecutionEngineDumpToObjectFileWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirExecutionEngineRegisterSymbolWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirExecutionEngineRegisterSymbol(arg0, arg1, arg2);
}

export fn fizz_nif_mlirExecutionEngineRegisterSymbol(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirExecutionEngine = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirExecutionEngine, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirExecutionEngineRegisterSymbolWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirExecutionEngineLookupWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirExecutionEngineLookup(arg0, arg1);
}

export fn fizz_nif_mlirExecutionEngineLookup(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirExecutionEngine = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirExecutionEngine, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void_ptr, @sizeOf(?*anyopaque));

  const RType = ?*anyopaque;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirExecutionEngineLookupWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirExecutionEngineLookupPackedWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirExecutionEngineLookupPacked(arg0, arg1);
}

export fn fizz_nif_mlirExecutionEngineLookupPacked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirExecutionEngine = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirExecutionEngine, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void_ptr, @sizeOf(?*anyopaque));

  const RType = ?*anyopaque;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirExecutionEngineLookupPackedWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirExecutionEngineInvokePackedWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirExecutionEngineInvokePacked(arg0, arg1, arg2);
}

export fn fizz_nif_mlirExecutionEngineInvokePacked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirExecutionEngine = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirExecutionEngine, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: [*c]?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr__nullable__pointer_anyopaque, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLogicalResult, @sizeOf(c.struct_MlirLogicalResult));

  const RType = c.struct_MlirLogicalResult;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirExecutionEngineInvokePackedWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirExecutionEngineDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirExecutionEngineDestroy(arg0);
}

export fn fizz_nif_mlirExecutionEngineDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirExecutionEngine = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirExecutionEngine, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirExecutionEngineDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirExecutionEngineCreateWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirExecutionEngineCreate(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirExecutionEngineCreate(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirModule = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirModule, args[0]);
  var arg1: c_int = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_int, args[1]);
  var arg2: c_int = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_int, args[2]);
  var arg3: [*c]const c.struct_MlirStringRef = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type__c_ptr_const_c_struct_MlirStringRef, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirExecutionEngine, @sizeOf(c.struct_MlirExecutionEngine));

  const RType = c.struct_MlirExecutionEngine;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirExecutionEngineCreateWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__tensor__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__tensor__();
}

export fn fizz_nif_mlirGetDialectHandle__tensor__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__tensor__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterSparseTensorSparsificationWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterSparseTensorSparsification();
}

export fn fizz_nif_mlirRegisterSparseTensorSparsification(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterSparseTensorSparsificationWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateSparseTensorSparsificationWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateSparseTensorSparsification();
}

export fn fizz_nif_mlirCreateSparseTensorSparsification(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateSparseTensorSparsificationWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterSparseTensorSparseTensorConversionWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterSparseTensorSparseTensorConversion();
}

export fn fizz_nif_mlirRegisterSparseTensorSparseTensorConversion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterSparseTensorSparseTensorConversionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateSparseTensorSparseTensorConversionWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateSparseTensorSparseTensorConversion();
}

export fn fizz_nif_mlirCreateSparseTensorSparseTensorConversion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateSparseTensorSparseTensorConversionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterSparseTensorPassesWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterSparseTensorPasses();
}

export fn fizz_nif_mlirRegisterSparseTensorPasses(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterSparseTensorPassesWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseTensorEncodingAttrGetIndexBitWidthWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSparseTensorEncodingAttrGetIndexBitWidth(arg0);
}

export fn fizz_nif_mlirSparseTensorEncodingAttrGetIndexBitWidth(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_int, @sizeOf(c_int));

  const RType = c_int;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseTensorEncodingAttrGetIndexBitWidthWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseTensorEncodingAttrGetPointerBitWidthWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSparseTensorEncodingAttrGetPointerBitWidth(arg0);
}

export fn fizz_nif_mlirSparseTensorEncodingAttrGetPointerBitWidth(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_int, @sizeOf(c_int));

  const RType = c_int;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseTensorEncodingAttrGetPointerBitWidthWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseTensorEncodingAttrGetDimOrderingWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSparseTensorEncodingAttrGetDimOrdering(arg0);
}

export fn fizz_nif_mlirSparseTensorEncodingAttrGetDimOrdering(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseTensorEncodingAttrGetDimOrderingWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseTensorEncodingAttrGetDimLevelTypeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirSparseTensorEncodingAttrGetDimLevelType(arg0, arg1);
}

export fn fizz_nif_mlirSparseTensorEncodingAttrGetDimLevelType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_uint, @sizeOf(c_uint));

  const RType = c_uint;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseTensorEncodingAttrGetDimLevelTypeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseTensorEncodingGetNumDimLevelTypesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSparseTensorEncodingGetNumDimLevelTypes(arg0);
}

export fn fizz_nif_mlirSparseTensorEncodingGetNumDimLevelTypes(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseTensorEncodingGetNumDimLevelTypesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseTensorEncodingAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype, arg5: anytype) void {
  ret.* = c.mlirSparseTensorEncodingAttrGet(arg0, arg1, arg2, arg3, arg4, arg5);
}

export fn fizz_nif_mlirSparseTensorEncodingAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c_uint = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_uint, args[2]);
  var arg3: c.struct_MlirAffineMap = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_struct_MlirAffineMap, args[3]);
  var arg4: c_int = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_c_int, args[4]);
  var arg5: c_int = undefined; arg5 = beam.fetch_resource(arg5, env, resource_type_c_int, args[5]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseTensorEncodingAttrGetWrapper(obj, arg0, arg1, arg2, arg3, arg4, arg5);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsASparseTensorEncodingAttrWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsASparseTensorEncodingAttr(arg0);
}

export fn fizz_nif_mlirAttributeIsASparseTensorEncodingAttr(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsASparseTensorEncodingAttrWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__sparse_tensor__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__sparse_tensor__();
}

export fn fizz_nif_mlirGetDialectHandle__sparse_tensor__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__sparse_tensor__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__shape__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__shape__();
}

export fn fizz_nif_mlirGetDialectHandle__shape__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__shape__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__scf__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__scf__();
}

export fn fizz_nif_mlirGetDialectHandle__scf__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__scf__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCalibratedQuantizedTypeGetMaxWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirCalibratedQuantizedTypeGetMax(arg0);
}

export fn fizz_nif_mlirCalibratedQuantizedTypeGetMax(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f64, @sizeOf(f64));

  const RType = f64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCalibratedQuantizedTypeGetMaxWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirCalibratedQuantizedTypeGetMinWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirCalibratedQuantizedTypeGetMin(arg0);
}

export fn fizz_nif_mlirCalibratedQuantizedTypeGetMin(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f64, @sizeOf(f64));

  const RType = f64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCalibratedQuantizedTypeGetMinWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirCalibratedQuantizedTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirCalibratedQuantizedTypeGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirCalibratedQuantizedTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: f64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_f64, args[1]);
  var arg2: f64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_f64, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCalibratedQuantizedTypeGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsACalibratedQuantizedTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsACalibratedQuantizedType(arg0);
}

export fn fizz_nif_mlirTypeIsACalibratedQuantizedType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsACalibratedQuantizedTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedPerAxisTypeIsFixedPointWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUniformQuantizedPerAxisTypeIsFixedPoint(arg0);
}

export fn fizz_nif_mlirUniformQuantizedPerAxisTypeIsFixedPoint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedPerAxisTypeIsFixedPointWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedPerAxisTypeGetQuantizedDimensionWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(arg0);
}

export fn fizz_nif_mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i32, @sizeOf(i32));

  const RType = i32;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedPerAxisTypeGetQuantizedDimensionWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedPerAxisTypeGetZeroPointWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirUniformQuantizedPerAxisTypeGetZeroPoint(arg0, arg1);
}

export fn fizz_nif_mlirUniformQuantizedPerAxisTypeGetZeroPoint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedPerAxisTypeGetZeroPointWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedPerAxisTypeGetScaleWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirUniformQuantizedPerAxisTypeGetScale(arg0, arg1);
}

export fn fizz_nif_mlirUniformQuantizedPerAxisTypeGetScale(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f64, @sizeOf(f64));

  const RType = f64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedPerAxisTypeGetScaleWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedPerAxisTypeGetNumDimsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUniformQuantizedPerAxisTypeGetNumDims(arg0);
}

export fn fizz_nif_mlirUniformQuantizedPerAxisTypeGetNumDims(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedPerAxisTypeGetNumDimsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedPerAxisTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype, arg5: anytype, arg6: anytype, arg7: anytype, arg8: anytype) void {
  ret.* = c.mlirUniformQuantizedPerAxisTypeGet(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
}

export fn fizz_nif_mlirUniformQuantizedPerAxisTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c_uint = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_uint, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirType, args[2]);
  var arg3: isize = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_isize, args[3]);
  var arg4: [*c]f64 = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type__c_ptr_f64, args[4]);
  var arg5: [*c]i64 = undefined; arg5 = beam.fetch_resource(arg5, env, resource_type__c_ptr_i64, args[5]);
  var arg6: i32 = undefined; arg6 = beam.fetch_resource(arg6, env, resource_type_i32, args[6]);
  var arg7: i64 = undefined; arg7 = beam.fetch_resource(arg7, env, resource_type_i64, args[7]);
  var arg8: i64 = undefined; arg8 = beam.fetch_resource(arg8, env, resource_type_i64, args[8]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedPerAxisTypeGetWrapper(obj, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAUniformQuantizedPerAxisTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAUniformQuantizedPerAxisType(arg0);
}

export fn fizz_nif_mlirTypeIsAUniformQuantizedPerAxisType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAUniformQuantizedPerAxisTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedTypeIsFixedPointWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUniformQuantizedTypeIsFixedPoint(arg0);
}

export fn fizz_nif_mlirUniformQuantizedTypeIsFixedPoint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedTypeIsFixedPointWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedTypeGetZeroPointWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUniformQuantizedTypeGetZeroPoint(arg0);
}

export fn fizz_nif_mlirUniformQuantizedTypeGetZeroPoint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedTypeGetZeroPointWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedTypeGetScaleWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUniformQuantizedTypeGetScale(arg0);
}

export fn fizz_nif_mlirUniformQuantizedTypeGetScale(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f64, @sizeOf(f64));

  const RType = f64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedTypeGetScaleWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUniformQuantizedTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype, arg5: anytype, arg6: anytype) void {
  ret.* = c.mlirUniformQuantizedTypeGet(arg0, arg1, arg2, arg3, arg4, arg5, arg6);
}

export fn fizz_nif_mlirUniformQuantizedTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c_uint = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_uint, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirType, args[2]);
  var arg3: f64 = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_f64, args[3]);
  var arg4: i64 = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_i64, args[4]);
  var arg5: i64 = undefined; arg5 = beam.fetch_resource(arg5, env, resource_type_i64, args[5]);
  var arg6: i64 = undefined; arg6 = beam.fetch_resource(arg6, env, resource_type_i64, args[6]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUniformQuantizedTypeGetWrapper(obj, arg0, arg1, arg2, arg3, arg4, arg5, arg6);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAUniformQuantizedTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAUniformQuantizedType(arg0);
}

export fn fizz_nif_mlirTypeIsAUniformQuantizedType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAUniformQuantizedTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAnyQuantizedTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirAnyQuantizedTypeGet(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirAnyQuantizedTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c_uint = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_uint, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirType, args[2]);
  var arg3: i64 = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_i64, args[3]);
  var arg4: i64 = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_i64, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAnyQuantizedTypeGetWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAAnyQuantizedTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAAnyQuantizedType(arg0);
}

export fn fizz_nif_mlirTypeIsAAnyQuantizedType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAAnyQuantizedTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeCastExpressedToStorageTypeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirQuantizedTypeCastExpressedToStorageType(arg0, arg1);
}

export fn fizz_nif_mlirQuantizedTypeCastExpressedToStorageType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeCastExpressedToStorageTypeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeCastToExpressedTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeCastToExpressedType(arg0);
}

export fn fizz_nif_mlirQuantizedTypeCastToExpressedType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeCastToExpressedTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeCastFromExpressedTypeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirQuantizedTypeCastFromExpressedType(arg0, arg1);
}

export fn fizz_nif_mlirQuantizedTypeCastFromExpressedType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeCastFromExpressedTypeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeCastToStorageTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeCastToStorageType(arg0);
}

export fn fizz_nif_mlirQuantizedTypeCastToStorageType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeCastToStorageTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeCastFromStorageTypeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirQuantizedTypeCastFromStorageType(arg0, arg1);
}

export fn fizz_nif_mlirQuantizedTypeCastFromStorageType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeCastFromStorageTypeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetQuantizedElementTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeGetQuantizedElementType(arg0);
}

export fn fizz_nif_mlirQuantizedTypeGetQuantizedElementType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetQuantizedElementTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeIsCompatibleExpressedTypeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirQuantizedTypeIsCompatibleExpressedType(arg0, arg1);
}

export fn fizz_nif_mlirQuantizedTypeIsCompatibleExpressedType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeIsCompatibleExpressedTypeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetStorageTypeIntegralWidthWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeGetStorageTypeIntegralWidth(arg0);
}

export fn fizz_nif_mlirQuantizedTypeGetStorageTypeIntegralWidth(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_uint, @sizeOf(c_uint));

  const RType = c_uint;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetStorageTypeIntegralWidthWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetStorageTypeMaxWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeGetStorageTypeMax(arg0);
}

export fn fizz_nif_mlirQuantizedTypeGetStorageTypeMax(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetStorageTypeMaxWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetStorageTypeMinWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeGetStorageTypeMin(arg0);
}

export fn fizz_nif_mlirQuantizedTypeGetStorageTypeMin(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetStorageTypeMinWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetStorageTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeGetStorageType(arg0);
}

export fn fizz_nif_mlirQuantizedTypeGetStorageType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetStorageTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeIsSignedWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeIsSigned(arg0);
}

export fn fizz_nif_mlirQuantizedTypeIsSigned(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeIsSignedWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetFlagsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeGetFlags(arg0);
}

export fn fizz_nif_mlirQuantizedTypeGetFlags(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_uint, @sizeOf(c_uint));

  const RType = c_uint;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetFlagsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetExpressedTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirQuantizedTypeGetExpressedType(arg0);
}

export fn fizz_nif_mlirQuantizedTypeGetExpressedType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetExpressedTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetDefaultMaximumForIntegerWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirQuantizedTypeGetDefaultMaximumForInteger(arg0, arg1);
}

export fn fizz_nif_mlirQuantizedTypeGetDefaultMaximumForInteger(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: bool = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_bool, args[0]);
  var arg1: c_uint = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_uint, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetDefaultMaximumForIntegerWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetDefaultMinimumForIntegerWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirQuantizedTypeGetDefaultMinimumForInteger(arg0, arg1);
}

export fn fizz_nif_mlirQuantizedTypeGetDefaultMinimumForInteger(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: bool = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_bool, args[0]);
  var arg1: c_uint = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_uint, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetDefaultMinimumForIntegerWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirQuantizedTypeGetSignedFlagWrapper(ret: anytype, ) void {
  ret.* = c.mlirQuantizedTypeGetSignedFlag();
}

export fn fizz_nif_mlirQuantizedTypeGetSignedFlag(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_uint, @sizeOf(c_uint));

  const RType = c_uint;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirQuantizedTypeGetSignedFlagWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAQuantizedTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAQuantizedType(arg0);
}

export fn fizz_nif_mlirTypeIsAQuantizedType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAQuantizedTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__quant__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__quant__();
}

export fn fizz_nif_mlirGetDialectHandle__quant__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__quant__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirPDLValueTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPDLValueTypeGet(arg0);
}

export fn fizz_nif_mlirPDLValueTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPDLValueTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAPDLValueTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAPDLValueType(arg0);
}

export fn fizz_nif_mlirTypeIsAPDLValueType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAPDLValueTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirPDLTypeTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPDLTypeTypeGet(arg0);
}

export fn fizz_nif_mlirPDLTypeTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPDLTypeTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAPDLTypeTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAPDLTypeType(arg0);
}

export fn fizz_nif_mlirTypeIsAPDLTypeType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAPDLTypeTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirPDLRangeTypeGetElementTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPDLRangeTypeGetElementType(arg0);
}

export fn fizz_nif_mlirPDLRangeTypeGetElementType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPDLRangeTypeGetElementTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirPDLRangeTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPDLRangeTypeGet(arg0);
}

export fn fizz_nif_mlirPDLRangeTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPDLRangeTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAPDLRangeTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAPDLRangeType(arg0);
}

export fn fizz_nif_mlirTypeIsAPDLRangeType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAPDLRangeTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirPDLOperationTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPDLOperationTypeGet(arg0);
}

export fn fizz_nif_mlirPDLOperationTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPDLOperationTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAPDLOperationTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAPDLOperationType(arg0);
}

export fn fizz_nif_mlirTypeIsAPDLOperationType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAPDLOperationTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirPDLAttributeTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPDLAttributeTypeGet(arg0);
}

export fn fizz_nif_mlirPDLAttributeTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPDLAttributeTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAPDLAttributeTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAPDLAttributeType(arg0);
}

export fn fizz_nif_mlirTypeIsAPDLAttributeType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAPDLAttributeTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAPDLTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAPDLType(arg0);
}

export fn fizz_nif_mlirTypeIsAPDLType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAPDLTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__pdl__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__pdl__();
}

export fn fizz_nif_mlirGetDialectHandle__pdl__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__pdl__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgTilingWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgTiling();
}

export fn fizz_nif_mlirRegisterLinalgLinalgTiling(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgTilingWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgTilingWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgTiling();
}

export fn fizz_nif_mlirCreateLinalgLinalgTiling(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgTilingWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyVectorizePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyVectorizePass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyVectorizePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyVectorizePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyVectorizePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyVectorizePass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyVectorizePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyVectorizePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyTilePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyTilePass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyTilePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyTilePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyTilePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyTilePass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyTilePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyTilePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyTileAndFusePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyTileAndFusePass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyTileAndFusePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyTileAndFusePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyTileAndFusePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyTileAndFusePass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyTileAndFusePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyTileAndFusePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyRemoveMarkersPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyRemoveMarkersPass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyRemoveMarkersPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyRemoveMarkersPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyRemoveMarkersPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyRemoveMarkersPass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyRemoveMarkersPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyRemoveMarkersPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyPromotePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyPromotePass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyPromotePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyPromotePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyPromotePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyPromotePass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyPromotePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyPromotePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyPeelPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyPeelPass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyPeelPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyPeelPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyPeelPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyPeelPass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyPeelPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyPeelPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyPadPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyPadPass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyPadPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyPadPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyPadPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyPadPass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyPadPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyPadPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyLowerVectorsPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyLowerVectorsPass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyLowerVectorsPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyLowerVectorsPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyLowerVectorsPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyLowerVectorsPass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyLowerVectorsPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyLowerVectorsPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyInterchangePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyInterchangePass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyInterchangePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyInterchangePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyInterchangePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyInterchangePass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyInterchangePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyInterchangePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyGeneralizePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyGeneralizePass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyGeneralizePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyGeneralizePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyGeneralizePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyGeneralizePass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyGeneralizePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyGeneralizePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyEnablePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyEnablePass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyEnablePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyEnablePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyEnablePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyEnablePass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyEnablePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyEnablePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgStrategyDecomposePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgStrategyDecomposePass();
}

export fn fizz_nif_mlirRegisterLinalgLinalgStrategyDecomposePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgStrategyDecomposePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgStrategyDecomposePassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgStrategyDecomposePass();
}

export fn fizz_nif_mlirCreateLinalgLinalgStrategyDecomposePass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgStrategyDecomposePassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgPromotionWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgPromotion();
}

export fn fizz_nif_mlirRegisterLinalgLinalgPromotion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgPromotionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgPromotionWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgPromotion();
}

export fn fizz_nif_mlirCreateLinalgLinalgPromotion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgPromotionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgNamedOpConversionWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgNamedOpConversion();
}

export fn fizz_nif_mlirRegisterLinalgLinalgNamedOpConversion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgNamedOpConversionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgNamedOpConversionWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgNamedOpConversion();
}

export fn fizz_nif_mlirCreateLinalgLinalgNamedOpConversion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgNamedOpConversionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgLowerToParallelLoopsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgLowerToParallelLoops();
}

export fn fizz_nif_mlirRegisterLinalgLinalgLowerToParallelLoops(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgLowerToParallelLoopsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgLowerToParallelLoopsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgLowerToParallelLoops();
}

export fn fizz_nif_mlirCreateLinalgLinalgLowerToParallelLoops(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgLowerToParallelLoopsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgLowerToLoopsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgLowerToLoops();
}

export fn fizz_nif_mlirRegisterLinalgLinalgLowerToLoops(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgLowerToLoopsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgLowerToLoopsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgLowerToLoops();
}

export fn fizz_nif_mlirCreateLinalgLinalgLowerToLoops(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgLowerToLoopsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgLowerToAffineLoopsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgLowerToAffineLoops();
}

export fn fizz_nif_mlirRegisterLinalgLinalgLowerToAffineLoops(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgLowerToAffineLoopsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgLowerToAffineLoopsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgLowerToAffineLoops();
}

export fn fizz_nif_mlirCreateLinalgLinalgLowerToAffineLoops(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgLowerToAffineLoopsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgInlineScalarOperandsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgInlineScalarOperands();
}

export fn fizz_nif_mlirRegisterLinalgLinalgInlineScalarOperands(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgInlineScalarOperandsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgInlineScalarOperandsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgInlineScalarOperands();
}

export fn fizz_nif_mlirCreateLinalgLinalgInlineScalarOperands(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgInlineScalarOperandsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgInitTensorToAllocTensorWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgInitTensorToAllocTensor();
}

export fn fizz_nif_mlirRegisterLinalgLinalgInitTensorToAllocTensor(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgInitTensorToAllocTensorWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgInitTensorToAllocTensorWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgInitTensorToAllocTensor();
}

export fn fizz_nif_mlirCreateLinalgLinalgInitTensorToAllocTensor(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgInitTensorToAllocTensorWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgGeneralizationWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgGeneralization();
}

export fn fizz_nif_mlirRegisterLinalgLinalgGeneralization(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgGeneralizationWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgGeneralizationWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgGeneralization();
}

export fn fizz_nif_mlirCreateLinalgLinalgGeneralization(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgGeneralizationWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgFoldUnitExtentDimsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgFoldUnitExtentDims();
}

export fn fizz_nif_mlirRegisterLinalgLinalgFoldUnitExtentDims(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgFoldUnitExtentDimsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgFoldUnitExtentDimsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgFoldUnitExtentDims();
}

export fn fizz_nif_mlirCreateLinalgLinalgFoldUnitExtentDims(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgFoldUnitExtentDimsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgElementwiseOpFusionWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgElementwiseOpFusion();
}

export fn fizz_nif_mlirRegisterLinalgLinalgElementwiseOpFusion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgElementwiseOpFusionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgElementwiseOpFusionWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgElementwiseOpFusion();
}

export fn fizz_nif_mlirCreateLinalgLinalgElementwiseOpFusion(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgElementwiseOpFusionWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgDetensorizeWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgDetensorize();
}

export fn fizz_nif_mlirRegisterLinalgLinalgDetensorize(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgDetensorizeWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgDetensorizeWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgDetensorize();
}

export fn fizz_nif_mlirCreateLinalgLinalgDetensorize(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgDetensorizeWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgLinalgBufferizeWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgLinalgBufferize();
}

export fn fizz_nif_mlirRegisterLinalgLinalgBufferize(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgLinalgBufferizeWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgLinalgBufferizeWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgLinalgBufferize();
}

export fn fizz_nif_mlirCreateLinalgLinalgBufferize(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgLinalgBufferizeWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgConvertElementwiseToLinalgWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgConvertElementwiseToLinalg();
}

export fn fizz_nif_mlirRegisterLinalgConvertElementwiseToLinalg(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgConvertElementwiseToLinalgWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateLinalgConvertElementwiseToLinalgWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateLinalgConvertElementwiseToLinalg();
}

export fn fizz_nif_mlirCreateLinalgConvertElementwiseToLinalg(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateLinalgConvertElementwiseToLinalgWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterLinalgPassesWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterLinalgPasses();
}

export fn fizz_nif_mlirRegisterLinalgPasses(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterLinalgPassesWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__linalg__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__linalg__();
}

export fn fizz_nif_mlirGetDialectHandle__linalg__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__linalg__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirLinalgFillBuiltinNamedOpRegionWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirLinalgFillBuiltinNamedOpRegion(arg0);
}

export fn fizz_nif_mlirLinalgFillBuiltinNamedOpRegion(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLinalgFillBuiltinNamedOpRegionWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirLLVMStructTypeLiteralGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirLLVMStructTypeLiteralGet(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirLLVMStructTypeLiteralGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirType, args[2]);
  var arg3: bool = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_bool, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLLVMStructTypeLiteralGetWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirLLVMFunctionTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirLLVMFunctionTypeGet(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirLLVMFunctionTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirType, args[2]);
  var arg3: bool = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_bool, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLLVMFunctionTypeGetWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirLLVMArrayTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirLLVMArrayTypeGet(arg0, arg1);
}

export fn fizz_nif_mlirLLVMArrayTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c_uint = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_uint, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLLVMArrayTypeGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirLLVMVoidTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirLLVMVoidTypeGet(arg0);
}

export fn fizz_nif_mlirLLVMVoidTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLLVMVoidTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirLLVMPointerTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirLLVMPointerTypeGet(arg0, arg1);
}

export fn fizz_nif_mlirLLVMPointerTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c_uint = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_uint, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLLVMPointerTypeGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__llvm__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__llvm__();
}

export fn fizz_nif_mlirGetDialectHandle__llvm__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__llvm__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterGPUGpuMapParallelLoopsPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterGPUGpuMapParallelLoopsPass();
}

export fn fizz_nif_mlirRegisterGPUGpuMapParallelLoopsPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterGPUGpuMapParallelLoopsPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateGPUGpuMapParallelLoopsPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateGPUGpuMapParallelLoopsPass();
}

export fn fizz_nif_mlirCreateGPUGpuMapParallelLoopsPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateGPUGpuMapParallelLoopsPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterGPUGpuLaunchSinkIndexComputationsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterGPUGpuLaunchSinkIndexComputations();
}

export fn fizz_nif_mlirRegisterGPUGpuLaunchSinkIndexComputations(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterGPUGpuLaunchSinkIndexComputationsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateGPUGpuLaunchSinkIndexComputationsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateGPUGpuLaunchSinkIndexComputations();
}

export fn fizz_nif_mlirCreateGPUGpuLaunchSinkIndexComputations(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateGPUGpuLaunchSinkIndexComputationsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterGPUGpuKernelOutliningWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterGPUGpuKernelOutlining();
}

export fn fizz_nif_mlirRegisterGPUGpuKernelOutlining(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterGPUGpuKernelOutliningWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateGPUGpuKernelOutliningWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateGPUGpuKernelOutlining();
}

export fn fizz_nif_mlirCreateGPUGpuKernelOutlining(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateGPUGpuKernelOutliningWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterGPUGpuAsyncRegionPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterGPUGpuAsyncRegionPass();
}

export fn fizz_nif_mlirRegisterGPUGpuAsyncRegionPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterGPUGpuAsyncRegionPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateGPUGpuAsyncRegionPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateGPUGpuAsyncRegionPass();
}

export fn fizz_nif_mlirCreateGPUGpuAsyncRegionPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateGPUGpuAsyncRegionPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterGPUPassesWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterGPUPasses();
}

export fn fizz_nif_mlirRegisterGPUPasses(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterGPUPassesWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__gpu__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__gpu__();
}

export fn fizz_nif_mlirGetDialectHandle__gpu__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__gpu__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__func__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__func__();
}

export fn fizz_nif_mlirGetDialectHandle__func__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__func__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__elixir__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__elixir__();
}

export fn fizz_nif_mlirGetDialectHandle__elixir__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__elixir__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__cf__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__cf__();
}

export fn fizz_nif_mlirGetDialectHandle__cf__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__cf__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAsyncAsyncToAsyncRuntimeWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterAsyncAsyncToAsyncRuntime();
}

export fn fizz_nif_mlirRegisterAsyncAsyncToAsyncRuntime(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAsyncAsyncToAsyncRuntimeWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateAsyncAsyncToAsyncRuntimeWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateAsyncAsyncToAsyncRuntime();
}

export fn fizz_nif_mlirCreateAsyncAsyncToAsyncRuntime(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateAsyncAsyncToAsyncRuntimeWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAsyncAsyncRuntimeRefCountingOptWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterAsyncAsyncRuntimeRefCountingOpt();
}

export fn fizz_nif_mlirRegisterAsyncAsyncRuntimeRefCountingOpt(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAsyncAsyncRuntimeRefCountingOptWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateAsyncAsyncRuntimeRefCountingOptWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateAsyncAsyncRuntimeRefCountingOpt();
}

export fn fizz_nif_mlirCreateAsyncAsyncRuntimeRefCountingOpt(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateAsyncAsyncRuntimeRefCountingOptWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAsyncAsyncRuntimeRefCountingWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterAsyncAsyncRuntimeRefCounting();
}

export fn fizz_nif_mlirRegisterAsyncAsyncRuntimeRefCounting(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAsyncAsyncRuntimeRefCountingWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateAsyncAsyncRuntimeRefCountingWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateAsyncAsyncRuntimeRefCounting();
}

export fn fizz_nif_mlirCreateAsyncAsyncRuntimeRefCounting(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateAsyncAsyncRuntimeRefCountingWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAsyncAsyncRuntimePolicyBasedRefCountingWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting();
}

export fn fizz_nif_mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAsyncAsyncRuntimePolicyBasedRefCountingWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateAsyncAsyncRuntimePolicyBasedRefCountingWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting();
}

export fn fizz_nif_mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateAsyncAsyncRuntimePolicyBasedRefCountingWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAsyncAsyncParallelForWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterAsyncAsyncParallelFor();
}

export fn fizz_nif_mlirRegisterAsyncAsyncParallelFor(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAsyncAsyncParallelForWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateAsyncAsyncParallelForWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateAsyncAsyncParallelFor();
}

export fn fizz_nif_mlirCreateAsyncAsyncParallelFor(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateAsyncAsyncParallelForWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAsyncPassesWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterAsyncPasses();
}

export fn fizz_nif_mlirRegisterAsyncPasses(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAsyncPassesWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirGetDialectHandle__async__Wrapper(ret: anytype, ) void {
  ret.* = c.mlirGetDialectHandle__async__();
}

export fn fizz_nif_mlirGetDialectHandle__async__(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectHandle, @sizeOf(c.struct_MlirDialectHandle));

  const RType = c.struct_MlirDialectHandle;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirGetDialectHandle__async__Wrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirEmitErrorWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirEmitError(arg0, arg1);
}

export fn fizz_nif_mlirEmitError(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: [*c]const u8 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__c_ptr_const_u8, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirEmitErrorWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextDetachDiagnosticHandlerWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirContextDetachDiagnosticHandler(arg0, arg1);
}

export fn fizz_nif_mlirContextDetachDiagnosticHandler(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: u64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_u64, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextDetachDiagnosticHandlerWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextAttachDiagnosticHandlerWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirContextAttachDiagnosticHandler(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirContextAttachDiagnosticHandler(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: ?fn(c.struct_MlirDiagnostic, ?*anyopaque) callconv(.C) c.struct_MlirLogicalResult = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirDiagnostic____nullable__pointer_anyopaque__callconv__C__c_struct_MlirLogicalResult, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);
  var arg3: ?fn(?*anyopaque) callconv(.C) void = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type__nullable_fn__nullable__pointer_anyopaque__callconv__C__void, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u64, @sizeOf(u64));

  const RType = u64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextAttachDiagnosticHandlerWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirDiagnosticGetNoteWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDiagnosticGetNote(arg0, arg1);
}

export fn fizz_nif_mlirDiagnosticGetNote(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDiagnostic = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDiagnostic, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDiagnostic, @sizeOf(c.struct_MlirDiagnostic));

  const RType = c.struct_MlirDiagnostic;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDiagnosticGetNoteWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDiagnosticGetNumNotesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDiagnosticGetNumNotes(arg0);
}

export fn fizz_nif_mlirDiagnosticGetNumNotes(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDiagnostic = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDiagnostic, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDiagnosticGetNumNotesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDiagnosticGetSeverityWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDiagnosticGetSeverity(arg0);
}

export fn fizz_nif_mlirDiagnosticGetSeverity(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDiagnostic = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDiagnostic, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_uint, @sizeOf(c_uint));

  const RType = c_uint;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDiagnosticGetSeverityWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDiagnosticGetLocationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDiagnosticGetLocation(arg0);
}

export fn fizz_nif_mlirDiagnosticGetLocation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDiagnostic = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDiagnostic, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLocation, @sizeOf(c.struct_MlirLocation));

  const RType = c.struct_MlirLocation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDiagnosticGetLocationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDiagnosticPrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDiagnosticPrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDiagnosticPrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDiagnostic = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDiagnostic, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDiagnosticPrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirIsGlobalDebugEnabledWrapper(ret: anytype, ) void {
  ret.* = c.mlirIsGlobalDebugEnabled();
}

export fn fizz_nif_mlirIsGlobalDebugEnabled(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIsGlobalDebugEnabledWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirEnableGlobalDebugWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirEnableGlobalDebug(arg0);
}

export fn fizz_nif_mlirEnableGlobalDebug(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: bool = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_bool, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirEnableGlobalDebugWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionTosaToTensorWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionTosaToTensor();
}

export fn fizz_nif_mlirRegisterConversionTosaToTensor(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionTosaToTensorWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionTosaToTensorWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionTosaToTensor();
}

export fn fizz_nif_mlirCreateConversionTosaToTensor(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionTosaToTensorWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionTosaToSCFWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionTosaToSCF();
}

export fn fizz_nif_mlirRegisterConversionTosaToSCF(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionTosaToSCFWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionTosaToSCFWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionTosaToSCF();
}

export fn fizz_nif_mlirCreateConversionTosaToSCF(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionTosaToSCFWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionTosaToLinalgNamedWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionTosaToLinalgNamed();
}

export fn fizz_nif_mlirRegisterConversionTosaToLinalgNamed(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionTosaToLinalgNamedWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionTosaToLinalgNamedWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionTosaToLinalgNamed();
}

export fn fizz_nif_mlirCreateConversionTosaToLinalgNamed(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionTosaToLinalgNamedWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionTosaToLinalgWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionTosaToLinalg();
}

export fn fizz_nif_mlirRegisterConversionTosaToLinalg(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionTosaToLinalgWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionTosaToLinalgWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionTosaToLinalg();
}

export fn fizz_nif_mlirCreateConversionTosaToLinalg(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionTosaToLinalgWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionTosaToArithWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionTosaToArith();
}

export fn fizz_nif_mlirRegisterConversionTosaToArith(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionTosaToArithWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionTosaToArithWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionTosaToArith();
}

export fn fizz_nif_mlirCreateConversionTosaToArith(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionTosaToArithWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionSCFToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionSCFToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionSCFToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionSCFToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionSCFToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionSCFToSPIRV();
}

export fn fizz_nif_mlirCreateConversionSCFToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionSCFToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionSCFToControlFlowWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionSCFToControlFlow();
}

export fn fizz_nif_mlirRegisterConversionSCFToControlFlow(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionSCFToControlFlowWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionSCFToControlFlowWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionSCFToControlFlow();
}

export fn fizz_nif_mlirCreateConversionSCFToControlFlow(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionSCFToControlFlowWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionReconcileUnrealizedCastsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionReconcileUnrealizedCasts();
}

export fn fizz_nif_mlirRegisterConversionReconcileUnrealizedCasts(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionReconcileUnrealizedCastsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionReconcileUnrealizedCastsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionReconcileUnrealizedCasts();
}

export fn fizz_nif_mlirCreateConversionReconcileUnrealizedCasts(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionReconcileUnrealizedCastsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionLowerHostCodeToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionLowerHostCodeToLLVM();
}

export fn fizz_nif_mlirRegisterConversionLowerHostCodeToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionLowerHostCodeToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionLowerHostCodeToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionLowerHostCodeToLLVM();
}

export fn fizz_nif_mlirCreateConversionLowerHostCodeToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionLowerHostCodeToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionGpuToLLVMConversionPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionGpuToLLVMConversionPass();
}

export fn fizz_nif_mlirRegisterConversionGpuToLLVMConversionPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionGpuToLLVMConversionPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionGpuToLLVMConversionPassWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionGpuToLLVMConversionPass();
}

export fn fizz_nif_mlirCreateConversionGpuToLLVMConversionPass(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionGpuToLLVMConversionPassWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCallsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCalls();
}

export fn fizz_nif_mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCalls(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCallsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertVulkanLaunchFuncToVulkanCallsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls();
}

export fn fizz_nif_mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertVulkanLaunchFuncToVulkanCallsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertVectorToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertVectorToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertVectorToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertVectorToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertVectorToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertVectorToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertVectorToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertVectorToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertVectorToSCFWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertVectorToSCF();
}

export fn fizz_nif_mlirRegisterConversionConvertVectorToSCF(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertVectorToSCFWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertVectorToSCFWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertVectorToSCF();
}

export fn fizz_nif_mlirCreateConversionConvertVectorToSCF(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertVectorToSCFWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertVectorToROCDLWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertVectorToROCDL();
}

export fn fizz_nif_mlirRegisterConversionConvertVectorToROCDL(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertVectorToROCDLWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertVectorToROCDLWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertVectorToROCDL();
}

export fn fizz_nif_mlirCreateConversionConvertVectorToROCDL(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertVectorToROCDLWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertVectorToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertVectorToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertVectorToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertVectorToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertVectorToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertVectorToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertVectorToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertVectorToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertVectorToGPUWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertVectorToGPU();
}

export fn fizz_nif_mlirRegisterConversionConvertVectorToGPU(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertVectorToGPUWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertVectorToGPUWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertVectorToGPU();
}

export fn fizz_nif_mlirCreateConversionConvertVectorToGPU(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertVectorToGPUWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertTensorToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertTensorToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertTensorToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertTensorToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertTensorToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertTensorToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertTensorToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertTensorToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertTensorToLinalgWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertTensorToLinalg();
}

export fn fizz_nif_mlirRegisterConversionConvertTensorToLinalg(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertTensorToLinalgWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertTensorToLinalgWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertTensorToLinalg();
}

export fn fizz_nif_mlirCreateConversionConvertTensorToLinalg(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertTensorToLinalgWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertShapeToStandardWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertShapeToStandard();
}

export fn fizz_nif_mlirRegisterConversionConvertShapeToStandard(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertShapeToStandardWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertShapeToStandardWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertShapeToStandard();
}

export fn fizz_nif_mlirCreateConversionConvertShapeToStandard(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertShapeToStandardWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertShapeConstraintsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertShapeConstraints();
}

export fn fizz_nif_mlirRegisterConversionConvertShapeConstraints(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertShapeConstraintsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertShapeConstraintsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertShapeConstraints();
}

export fn fizz_nif_mlirCreateConversionConvertShapeConstraints(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertShapeConstraintsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertSPIRVToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertSPIRVToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertSPIRVToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertSPIRVToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertSPIRVToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertSPIRVToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertSPIRVToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertSPIRVToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertSCFToOpenMPWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertSCFToOpenMP();
}

export fn fizz_nif_mlirRegisterConversionConvertSCFToOpenMP(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertSCFToOpenMPWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertSCFToOpenMPWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertSCFToOpenMP();
}

export fn fizz_nif_mlirCreateConversionConvertSCFToOpenMP(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertSCFToOpenMPWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertParallelLoopToGpuWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertParallelLoopToGpu();
}

export fn fizz_nif_mlirRegisterConversionConvertParallelLoopToGpu(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertParallelLoopToGpuWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertParallelLoopToGpuWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertParallelLoopToGpu();
}

export fn fizz_nif_mlirCreateConversionConvertParallelLoopToGpu(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertParallelLoopToGpuWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertPDLToPDLInterpWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertPDLToPDLInterp();
}

export fn fizz_nif_mlirRegisterConversionConvertPDLToPDLInterp(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertPDLToPDLInterpWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertPDLToPDLInterpWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertPDLToPDLInterp();
}

export fn fizz_nif_mlirCreateConversionConvertPDLToPDLInterp(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertPDLToPDLInterpWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertOpenMPToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertOpenMPToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertOpenMPToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertOpenMPToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertOpenMPToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertOpenMPToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertOpenMPToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertOpenMPToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertOpenACCToSCFWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertOpenACCToSCF();
}

export fn fizz_nif_mlirRegisterConversionConvertOpenACCToSCF(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertOpenACCToSCFWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertOpenACCToSCFWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertOpenACCToSCF();
}

export fn fizz_nif_mlirCreateConversionConvertOpenACCToSCF(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertOpenACCToSCFWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertOpenACCToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertOpenACCToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertOpenACCToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertOpenACCToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertOpenACCToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertOpenACCToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertOpenACCToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertOpenACCToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertNVGPUToNVVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertNVGPUToNVVM();
}

export fn fizz_nif_mlirRegisterConversionConvertNVGPUToNVVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertNVGPUToNVVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertNVGPUToNVVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertNVGPUToNVVM();
}

export fn fizz_nif_mlirCreateConversionConvertNVGPUToNVVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertNVGPUToNVVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertMemRefToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertMemRefToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertMemRefToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertMemRefToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertMemRefToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertMemRefToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertMemRefToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertMemRefToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertMemRefToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertMemRefToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertMemRefToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertMemRefToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertMemRefToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertMemRefToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertMemRefToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertMemRefToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertMathToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertMathToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertMathToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertMathToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertMathToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertMathToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertMathToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertMathToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertMathToLibmWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertMathToLibm();
}

export fn fizz_nif_mlirRegisterConversionConvertMathToLibm(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertMathToLibmWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertMathToLibmWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertMathToLibm();
}

export fn fizz_nif_mlirCreateConversionConvertMathToLibm(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertMathToLibmWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertMathToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertMathToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertMathToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertMathToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertMathToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertMathToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertMathToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertMathToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertLinalgToStandardWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertLinalgToStandard();
}

export fn fizz_nif_mlirRegisterConversionConvertLinalgToStandard(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertLinalgToStandardWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertLinalgToStandardWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertLinalgToStandard();
}

export fn fizz_nif_mlirCreateConversionConvertLinalgToStandard(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertLinalgToStandardWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertLinalgToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertLinalgToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertLinalgToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertLinalgToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertLinalgToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertLinalgToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertLinalgToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertLinalgToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertLinalgToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertLinalgToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertLinalgToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertLinalgToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertLinalgToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertLinalgToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertLinalgToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertLinalgToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertGpuOpsToROCDLOpsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertGpuOpsToROCDLOps();
}

export fn fizz_nif_mlirRegisterConversionConvertGpuOpsToROCDLOps(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertGpuOpsToROCDLOpsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertGpuOpsToROCDLOpsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertGpuOpsToROCDLOps();
}

export fn fizz_nif_mlirCreateConversionConvertGpuOpsToROCDLOps(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertGpuOpsToROCDLOpsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertGpuOpsToNVVMOpsWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertGpuOpsToNVVMOps();
}

export fn fizz_nif_mlirRegisterConversionConvertGpuOpsToNVVMOps(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertGpuOpsToNVVMOpsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertGpuOpsToNVVMOpsWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertGpuOpsToNVVMOps();
}

export fn fizz_nif_mlirCreateConversionConvertGpuOpsToNVVMOps(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertGpuOpsToNVVMOpsWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFuncWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc();
}

export fn fizz_nif_mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFuncWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFuncWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc();
}

export fn fizz_nif_mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFuncWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertGPUToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertGPUToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertGPUToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertGPUToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertGPUToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertGPUToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertGPUToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertGPUToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertFuncToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertFuncToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertFuncToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertFuncToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertFuncToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertFuncToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertFuncToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertFuncToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertFuncToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertFuncToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertFuncToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertFuncToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertFuncToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertFuncToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertFuncToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertFuncToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertControlFlowToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertControlFlowToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertControlFlowToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertControlFlowToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertControlFlowToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertControlFlowToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertControlFlowToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertControlFlowToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertControlFlowToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertControlFlowToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertControlFlowToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertControlFlowToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertControlFlowToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertControlFlowToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertControlFlowToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertControlFlowToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertComplexToStandardWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertComplexToStandard();
}

export fn fizz_nif_mlirRegisterConversionConvertComplexToStandard(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertComplexToStandardWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertComplexToStandardWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertComplexToStandard();
}

export fn fizz_nif_mlirCreateConversionConvertComplexToStandard(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertComplexToStandardWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertComplexToLibmWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertComplexToLibm();
}

export fn fizz_nif_mlirRegisterConversionConvertComplexToLibm(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertComplexToLibmWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertComplexToLibmWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertComplexToLibm();
}

export fn fizz_nif_mlirCreateConversionConvertComplexToLibm(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertComplexToLibmWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertComplexToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertComplexToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertComplexToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertComplexToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertComplexToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertComplexToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertComplexToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertComplexToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertBufferizationToMemRefWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertBufferizationToMemRef();
}

export fn fizz_nif_mlirRegisterConversionConvertBufferizationToMemRef(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertBufferizationToMemRefWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertBufferizationToMemRefWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertBufferizationToMemRef();
}

export fn fizz_nif_mlirCreateConversionConvertBufferizationToMemRef(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertBufferizationToMemRefWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertAsyncToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertAsyncToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertAsyncToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertAsyncToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertAsyncToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertAsyncToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertAsyncToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertAsyncToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertArmNeon2dToIntrWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertArmNeon2dToIntr();
}

export fn fizz_nif_mlirRegisterConversionConvertArmNeon2dToIntr(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertArmNeon2dToIntrWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertArmNeon2dToIntrWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertArmNeon2dToIntr();
}

export fn fizz_nif_mlirCreateConversionConvertArmNeon2dToIntr(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertArmNeon2dToIntrWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertArithmeticToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertArithmeticToSPIRV();
}

export fn fizz_nif_mlirRegisterConversionConvertArithmeticToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertArithmeticToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertArithmeticToSPIRVWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertArithmeticToSPIRV();
}

export fn fizz_nif_mlirCreateConversionConvertArithmeticToSPIRV(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertArithmeticToSPIRVWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertArithmeticToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertArithmeticToLLVM();
}

export fn fizz_nif_mlirRegisterConversionConvertArithmeticToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertArithmeticToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertArithmeticToLLVMWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertArithmeticToLLVM();
}

export fn fizz_nif_mlirCreateConversionConvertArithmeticToLLVM(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertArithmeticToLLVMWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertAffineToStandardWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertAffineToStandard();
}

export fn fizz_nif_mlirRegisterConversionConvertAffineToStandard(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertAffineToStandardWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertAffineToStandardWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertAffineToStandard();
}

export fn fizz_nif_mlirCreateConversionConvertAffineToStandard(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertAffineToStandardWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertAffineForToGPUWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertAffineForToGPU();
}

export fn fizz_nif_mlirRegisterConversionConvertAffineForToGPU(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertAffineForToGPUWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertAffineForToGPUWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertAffineForToGPU();
}

export fn fizz_nif_mlirCreateConversionConvertAffineForToGPU(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertAffineForToGPUWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionConvertAMDGPUToROCDLWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionConvertAMDGPUToROCDL();
}

export fn fizz_nif_mlirRegisterConversionConvertAMDGPUToROCDL(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionConvertAMDGPUToROCDLWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateConversionConvertAMDGPUToROCDLWrapper(ret: anytype, ) void {
  ret.* = c.mlirCreateConversionConvertAMDGPUToROCDL();
}

export fn fizz_nif_mlirCreateConversionConvertAMDGPUToROCDL(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateConversionConvertAMDGPUToROCDLWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterConversionPassesWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterConversionPasses();
}

export fn fizz_nif_mlirRegisterConversionPasses(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterConversionPassesWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirOpaqueTypeGetDataWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpaqueTypeGetData(arg0);
}

export fn fizz_nif_mlirOpaqueTypeGetData(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpaqueTypeGetDataWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpaqueTypeGetDialectNamespaceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpaqueTypeGetDialectNamespace(arg0);
}

export fn fizz_nif_mlirOpaqueTypeGetDialectNamespace(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpaqueTypeGetDialectNamespaceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpaqueTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOpaqueTypeGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOpaqueTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: c.struct_MlirStringRef = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirStringRef, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpaqueTypeGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAOpaqueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAOpaque(arg0);
}

export fn fizz_nif_mlirTypeIsAOpaque(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAOpaqueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirFunctionTypeGetResultWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirFunctionTypeGetResult(arg0, arg1);
}

export fn fizz_nif_mlirFunctionTypeGetResult(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFunctionTypeGetResultWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirFunctionTypeGetInputWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirFunctionTypeGetInput(arg0, arg1);
}

export fn fizz_nif_mlirFunctionTypeGetInput(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFunctionTypeGetInputWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirFunctionTypeGetNumResultsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirFunctionTypeGetNumResults(arg0);
}

export fn fizz_nif_mlirFunctionTypeGetNumResults(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFunctionTypeGetNumResultsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirFunctionTypeGetNumInputsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirFunctionTypeGetNumInputs(arg0);
}

export fn fizz_nif_mlirFunctionTypeGetNumInputs(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFunctionTypeGetNumInputsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirFunctionTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirFunctionTypeGet(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirFunctionTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirType, args[2]);
  var arg3: isize = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_isize, args[3]);
  var arg4: [*c]const c.struct_MlirType = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type__c_ptr_const_c_struct_MlirType, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFunctionTypeGetWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAFunctionWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAFunction(arg0);
}

export fn fizz_nif_mlirTypeIsAFunction(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAFunctionWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTupleTypeGetTypeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirTupleTypeGetType(arg0, arg1);
}

export fn fizz_nif_mlirTupleTypeGetType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTupleTypeGetTypeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirTupleTypeGetNumTypesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTupleTypeGetNumTypes(arg0);
}

export fn fizz_nif_mlirTupleTypeGetNumTypes(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTupleTypeGetNumTypesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTupleTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirTupleTypeGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirTupleTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirType, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTupleTypeGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsATupleWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsATuple(arg0);
}

export fn fizz_nif_mlirTypeIsATuple(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsATupleWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUnrankedMemrefGetMemorySpaceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUnrankedMemrefGetMemorySpace(arg0);
}

export fn fizz_nif_mlirUnrankedMemrefGetMemorySpace(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUnrankedMemrefGetMemorySpaceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirMemRefTypeGetMemorySpaceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirMemRefTypeGetMemorySpace(arg0);
}

export fn fizz_nif_mlirMemRefTypeGetMemorySpace(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirMemRefTypeGetMemorySpaceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirMemRefTypeGetAffineMapWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirMemRefTypeGetAffineMap(arg0);
}

export fn fizz_nif_mlirMemRefTypeGetAffineMap(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirMemRefTypeGetAffineMapWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirMemRefTypeGetLayoutWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirMemRefTypeGetLayout(arg0);
}

export fn fizz_nif_mlirMemRefTypeGetLayout(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirMemRefTypeGetLayoutWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUnrankedMemRefTypeGetCheckedWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirUnrankedMemRefTypeGetChecked(arg0, arg1, arg2);
}

export fn fizz_nif_mlirUnrankedMemRefTypeGetChecked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: c.struct_MlirAttribute = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirAttribute, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUnrankedMemRefTypeGetCheckedWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirUnrankedMemRefTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirUnrankedMemRefTypeGet(arg0, arg1);
}

export fn fizz_nif_mlirUnrankedMemRefTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirAttribute = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAttribute, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUnrankedMemRefTypeGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirMemRefTypeContiguousGetCheckedWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirMemRefTypeContiguousGetChecked(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirMemRefTypeContiguousGetChecked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);
  var arg3: [*c]const i64 = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type__c_ptr_const_i64, args[3]);
  var arg4: c.struct_MlirAttribute = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_c_struct_MlirAttribute, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirMemRefTypeContiguousGetCheckedWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirMemRefTypeContiguousGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirMemRefTypeContiguousGet(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirMemRefTypeContiguousGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const i64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_i64, args[2]);
  var arg3: c.struct_MlirAttribute = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_struct_MlirAttribute, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirMemRefTypeContiguousGetWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirMemRefTypeGetCheckedWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype, arg5: anytype) void {
  ret.* = c.mlirMemRefTypeGetChecked(arg0, arg1, arg2, arg3, arg4, arg5);
}

export fn fizz_nif_mlirMemRefTypeGetChecked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);
  var arg3: [*c]const i64 = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type__c_ptr_const_i64, args[3]);
  var arg4: c.struct_MlirAttribute = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_c_struct_MlirAttribute, args[4]);
  var arg5: c.struct_MlirAttribute = undefined; arg5 = beam.fetch_resource(arg5, env, resource_type_c_struct_MlirAttribute, args[5]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirMemRefTypeGetCheckedWrapper(obj, arg0, arg1, arg2, arg3, arg4, arg5);
  return e.enif_make_resource(env, ptr);
}

fn mlirMemRefTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirMemRefTypeGet(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirMemRefTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const i64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_i64, args[2]);
  var arg3: c.struct_MlirAttribute = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_struct_MlirAttribute, args[3]);
  var arg4: c.struct_MlirAttribute = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_c_struct_MlirAttribute, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirMemRefTypeGetWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAUnrankedMemRefWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAUnrankedMemRef(arg0);
}

export fn fizz_nif_mlirTypeIsAUnrankedMemRef(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAUnrankedMemRefWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAMemRefWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAMemRef(arg0);
}

export fn fizz_nif_mlirTypeIsAMemRef(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAMemRefWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUnrankedTensorTypeGetCheckedWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirUnrankedTensorTypeGetChecked(arg0, arg1);
}

export fn fizz_nif_mlirUnrankedTensorTypeGetChecked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUnrankedTensorTypeGetCheckedWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirUnrankedTensorTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUnrankedTensorTypeGet(arg0);
}

export fn fizz_nif_mlirUnrankedTensorTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUnrankedTensorTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirRankedTensorTypeGetEncodingWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirRankedTensorTypeGetEncoding(arg0);
}

export fn fizz_nif_mlirRankedTensorTypeGetEncoding(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRankedTensorTypeGetEncodingWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirRankedTensorTypeGetCheckedWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirRankedTensorTypeGetChecked(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirRankedTensorTypeGetChecked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const i64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_i64, args[2]);
  var arg3: c.struct_MlirType = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_struct_MlirType, args[3]);
  var arg4: c.struct_MlirAttribute = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_c_struct_MlirAttribute, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRankedTensorTypeGetCheckedWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirRankedTensorTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirRankedTensorTypeGet(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirRankedTensorTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: isize = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_isize, args[0]);
  var arg1: [*c]const i64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__c_ptr_const_i64, args[1]);
  var arg2: c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirType, args[2]);
  var arg3: c.struct_MlirAttribute = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_struct_MlirAttribute, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRankedTensorTypeGetWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAUnrankedTensorWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAUnrankedTensor(arg0);
}

export fn fizz_nif_mlirTypeIsAUnrankedTensor(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAUnrankedTensorWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsARankedTensorWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsARankedTensor(arg0);
}

export fn fizz_nif_mlirTypeIsARankedTensor(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsARankedTensorWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsATensorWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsATensor(arg0);
}

export fn fizz_nif_mlirTypeIsATensor(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsATensorWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirVectorTypeGetCheckedWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirVectorTypeGetChecked(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirVectorTypeGetChecked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const i64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_i64, args[2]);
  var arg3: c.struct_MlirType = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_struct_MlirType, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirVectorTypeGetCheckedWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirVectorTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirVectorTypeGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirVectorTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: isize = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_isize, args[0]);
  var arg1: [*c]const i64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__c_ptr_const_i64, args[1]);
  var arg2: c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirType, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirVectorTypeGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAVectorWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAVector(arg0);
}

export fn fizz_nif_mlirTypeIsAVector(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAVectorWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirShapedTypeIsDynamicStrideOrOffsetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirShapedTypeIsDynamicStrideOrOffset(arg0);
}

export fn fizz_nif_mlirShapedTypeIsDynamicStrideOrOffset(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: i64 = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_i64, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirShapedTypeIsDynamicStrideOrOffsetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirShapedTypeIsDynamicSizeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirShapedTypeIsDynamicSize(arg0);
}

export fn fizz_nif_mlirShapedTypeIsDynamicSize(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: i64 = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_i64, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirShapedTypeIsDynamicSizeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirShapedTypeGetDimSizeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirShapedTypeGetDimSize(arg0, arg1);
}

export fn fizz_nif_mlirShapedTypeGetDimSize(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirShapedTypeGetDimSizeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirShapedTypeIsDynamicDimWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirShapedTypeIsDynamicDim(arg0, arg1);
}

export fn fizz_nif_mlirShapedTypeIsDynamicDim(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirShapedTypeIsDynamicDimWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirShapedTypeHasStaticShapeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirShapedTypeHasStaticShape(arg0);
}

export fn fizz_nif_mlirShapedTypeHasStaticShape(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirShapedTypeHasStaticShapeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirShapedTypeGetRankWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirShapedTypeGetRank(arg0);
}

export fn fizz_nif_mlirShapedTypeGetRank(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirShapedTypeGetRankWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirShapedTypeHasRankWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirShapedTypeHasRank(arg0);
}

export fn fizz_nif_mlirShapedTypeHasRank(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirShapedTypeHasRankWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirShapedTypeGetElementTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirShapedTypeGetElementType(arg0);
}

export fn fizz_nif_mlirShapedTypeGetElementType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirShapedTypeGetElementTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAShapedWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAShaped(arg0);
}

export fn fizz_nif_mlirTypeIsAShaped(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAShapedWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirComplexTypeGetElementTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirComplexTypeGetElementType(arg0);
}

export fn fizz_nif_mlirComplexTypeGetElementType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirComplexTypeGetElementTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirComplexTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirComplexTypeGet(arg0);
}

export fn fizz_nif_mlirComplexTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirComplexTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAComplexWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAComplex(arg0);
}

export fn fizz_nif_mlirTypeIsAComplex(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAComplexWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirNoneTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirNoneTypeGet(arg0);
}

export fn fizz_nif_mlirNoneTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirNoneTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsANoneWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsANone(arg0);
}

export fn fizz_nif_mlirTypeIsANone(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsANoneWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirF64TypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirF64TypeGet(arg0);
}

export fn fizz_nif_mlirF64TypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirF64TypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAF64Wrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAF64(arg0);
}

export fn fizz_nif_mlirTypeIsAF64(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAF64Wrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirF32TypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirF32TypeGet(arg0);
}

export fn fizz_nif_mlirF32TypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirF32TypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAF32Wrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAF32(arg0);
}

export fn fizz_nif_mlirTypeIsAF32(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAF32Wrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirF16TypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirF16TypeGet(arg0);
}

export fn fizz_nif_mlirF16TypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirF16TypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAF16Wrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAF16(arg0);
}

export fn fizz_nif_mlirTypeIsAF16(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAF16Wrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBF16TypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBF16TypeGet(arg0);
}

export fn fizz_nif_mlirBF16TypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBF16TypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsABF16Wrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsABF16(arg0);
}

export fn fizz_nif_mlirTypeIsABF16(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsABF16Wrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIndexTypeGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIndexTypeGet(arg0);
}

export fn fizz_nif_mlirIndexTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIndexTypeGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAIndexWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAIndex(arg0);
}

export fn fizz_nif_mlirTypeIsAIndex(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAIndexWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerTypeIsUnsignedWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerTypeIsUnsigned(arg0);
}

export fn fizz_nif_mlirIntegerTypeIsUnsigned(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerTypeIsUnsignedWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerTypeIsSignedWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerTypeIsSigned(arg0);
}

export fn fizz_nif_mlirIntegerTypeIsSigned(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerTypeIsSignedWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerTypeIsSignlessWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerTypeIsSignless(arg0);
}

export fn fizz_nif_mlirIntegerTypeIsSignless(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerTypeIsSignlessWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerTypeGetWidthWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerTypeGetWidth(arg0);
}

export fn fizz_nif_mlirIntegerTypeGetWidth(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_uint, @sizeOf(c_uint));

  const RType = c_uint;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerTypeGetWidthWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerTypeUnsignedGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIntegerTypeUnsignedGet(arg0, arg1);
}

export fn fizz_nif_mlirIntegerTypeUnsignedGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c_uint = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_uint, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerTypeUnsignedGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerTypeSignedGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIntegerTypeSignedGet(arg0, arg1);
}

export fn fizz_nif_mlirIntegerTypeSignedGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c_uint = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_uint, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerTypeSignedGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerTypeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIntegerTypeGet(arg0, arg1);
}

export fn fizz_nif_mlirIntegerTypeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c_uint = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_uint, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerTypeGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIsAIntegerWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIsAInteger(arg0);
}

export fn fizz_nif_mlirTypeIsAInteger(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIsAIntegerWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseElementsAttrGetValuesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSparseElementsAttrGetValues(arg0);
}

export fn fizz_nif_mlirSparseElementsAttrGetValues(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseElementsAttrGetValuesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseElementsAttrGetIndicesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSparseElementsAttrGetIndices(arg0);
}

export fn fizz_nif_mlirSparseElementsAttrGetIndices(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseElementsAttrGetIndicesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSparseElementsAttributeWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirSparseElementsAttribute(arg0, arg1, arg2);
}

export fn fizz_nif_mlirSparseElementsAttribute(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirAttribute = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAttribute, args[1]);
  var arg2: c.struct_MlirAttribute = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirAttribute, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSparseElementsAttributeWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsASparseElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsASparseElements(arg0);
}

export fn fizz_nif_mlirAttributeIsASparseElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsASparseElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAOpaqueElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAOpaqueElements(arg0);
}

export fn fizz_nif_mlirAttributeIsAOpaqueElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAOpaqueElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetRawDataWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetRawData(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetRawData(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_const_void_ptr, @sizeOf(?*const anyopaque));

  const RType = ?*const anyopaque;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetRawDataWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetStringValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetStringValue(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetStringValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetStringValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetDoubleValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetDoubleValue(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetDoubleValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f64, @sizeOf(f64));

  const RType = f64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetDoubleValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetFloatValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetFloatValue(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetFloatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f32, @sizeOf(f32));

  const RType = f32;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetFloatValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetUInt64ValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetUInt64Value(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetUInt64Value(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u64, @sizeOf(u64));

  const RType = u64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetUInt64ValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetInt64ValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetInt64Value(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetInt64Value(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetInt64ValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetUInt32ValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetUInt32Value(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetUInt32Value(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u32, @sizeOf(u32));

  const RType = u32;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetUInt32ValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetInt32ValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetInt32Value(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetInt32Value(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i32, @sizeOf(i32));

  const RType = i32;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetInt32ValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetUInt16ValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetUInt16Value(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetUInt16Value(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u16, @sizeOf(u16));

  const RType = u16;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetUInt16ValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetInt16ValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetInt16Value(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetInt16Value(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i16, @sizeOf(i16));

  const RType = i16;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetInt16ValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetUInt8ValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetUInt8Value(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetUInt8Value(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u8, @sizeOf(u8));

  const RType = u8;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetUInt8ValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetInt8ValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetInt8Value(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetInt8Value(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i8, @sizeOf(i8));

  const RType = i8;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetInt8ValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetBoolValueWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetBoolValue(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrGetBoolValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetBoolValueWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetStringSplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetStringSplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetStringSplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetStringSplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetDoubleSplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetDoubleSplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetDoubleSplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f64, @sizeOf(f64));

  const RType = f64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetDoubleSplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetFloatSplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetFloatSplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetFloatSplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f32, @sizeOf(f32));

  const RType = f32;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetFloatSplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetUInt64SplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetUInt64SplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetUInt64SplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u64, @sizeOf(u64));

  const RType = u64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetUInt64SplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetInt64SplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetInt64SplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetInt64SplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetInt64SplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetUInt32SplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetUInt32SplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetUInt32SplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u32, @sizeOf(u32));

  const RType = u32;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetUInt32SplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetInt32SplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetInt32SplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetInt32SplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i32, @sizeOf(i32));

  const RType = i32;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetInt32SplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetUInt8SplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetUInt8SplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetUInt8SplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u8, @sizeOf(u8));

  const RType = u8;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetUInt8SplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetInt8SplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetInt8SplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetInt8SplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i8, @sizeOf(i8));

  const RType = i8;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetInt8SplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetBoolSplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetBoolSplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetBoolSplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_int, @sizeOf(c_int));

  const RType = c_int;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetBoolSplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetSplatValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrGetSplatValue(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrGetSplatValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetSplatValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrIsSplatWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDenseElementsAttrIsSplat(arg0);
}

export fn fizz_nif_mlirDenseElementsAttrIsSplat(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrIsSplatWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrReshapeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrReshapeGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrReshapeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrReshapeGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrStringGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrStringGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrStringGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]c.struct_MlirStringRef = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_c_struct_MlirStringRef, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrStringGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrBFloat16GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrBFloat16Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrBFloat16Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const u16 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_u16, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrBFloat16GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrDoubleGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrDoubleGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrDoubleGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const f64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_f64, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrDoubleGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrFloatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrFloatGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrFloatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const f32 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_f32, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrFloatGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrInt64GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrInt64Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrInt64Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const i64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_i64, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrInt64GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrUInt64GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrUInt64Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrUInt64Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const u64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_u64, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrUInt64GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrInt32GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrInt32Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrInt32Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const i32 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_i32, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrInt32GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrUInt32GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrUInt32Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrUInt32Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const u32 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_u32, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrUInt32GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrInt16GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrInt16Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrInt16Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const i16 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_i16, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrInt16GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrUInt16GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrUInt16Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrUInt16Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const u16 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_u16, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrUInt16GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrInt8GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrInt8Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrInt8Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const i8 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_i8, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrInt8GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrUInt8GetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrUInt8Get(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrUInt8Get(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const u8 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_u8, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrUInt8GetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrBoolGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrBoolGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrBoolGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c_int = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_int, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrBoolGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrDoubleSplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrDoubleSplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrDoubleSplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: f64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_f64, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrDoubleSplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrFloatSplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrFloatSplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrFloatSplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: f32 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_f32, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrFloatSplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrInt64SplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrInt64SplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrInt64SplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: i64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_i64, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrInt64SplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrUInt64SplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrUInt64SplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrUInt64SplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: u64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_u64, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrUInt64SplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrInt32SplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrInt32SplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrInt32SplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: i32 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_i32, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrInt32SplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrUInt32SplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrUInt32SplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrUInt32SplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: u32 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_u32, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrUInt32SplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrInt8SplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrInt8SplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrInt8SplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: i8 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_i8, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrInt8SplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrUInt8SplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrUInt8SplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrUInt8SplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: u8 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_u8, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrUInt8SplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrBoolSplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrBoolSplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrBoolSplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: bool = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_bool, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrBoolSplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrSplatGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDenseElementsAttrSplatGet(arg0, arg1);
}

export fn fizz_nif_mlirDenseElementsAttrSplatGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirAttribute = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAttribute, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrSplatGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrRawBufferGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrRawBufferGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrRawBufferGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: usize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_usize, args[1]);
  var arg2: ?*const anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_const_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrRawBufferGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirDenseElementsAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDenseElementsAttrGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDenseElementsAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirAttribute = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirAttribute, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDenseElementsAttrGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsADenseFPElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsADenseFPElements(arg0);
}

export fn fizz_nif_mlirAttributeIsADenseFPElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsADenseFPElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsADenseIntElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsADenseIntElements(arg0);
}

export fn fizz_nif_mlirAttributeIsADenseIntElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsADenseIntElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsADenseElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsADenseElements(arg0);
}

export fn fizz_nif_mlirAttributeIsADenseElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsADenseElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirElementsAttrGetNumElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirElementsAttrGetNumElements(arg0);
}

export fn fizz_nif_mlirElementsAttrGetNumElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirElementsAttrGetNumElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirElementsAttrIsValidIndexWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirElementsAttrIsValidIndex(arg0, arg1, arg2);
}

export fn fizz_nif_mlirElementsAttrIsValidIndex(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]u64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_u64, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirElementsAttrIsValidIndexWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirElementsAttrGetValueWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirElementsAttrGetValue(arg0, arg1, arg2);
}

export fn fizz_nif_mlirElementsAttrGetValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]u64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_u64, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirElementsAttrGetValueWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAElements(arg0);
}

export fn fizz_nif_mlirAttributeIsAElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirUnitAttrGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirUnitAttrGet(arg0);
}

export fn fizz_nif_mlirUnitAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirUnitAttrGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAUnitWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAUnit(arg0);
}

export fn fizz_nif_mlirAttributeIsAUnit(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAUnitWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeAttrGetValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeAttrGetValue(arg0);
}

export fn fizz_nif_mlirTypeAttrGetValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeAttrGetValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeAttrGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeAttrGet(arg0);
}

export fn fizz_nif_mlirTypeAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeAttrGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsATypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAType(arg0);
}

export fn fizz_nif_mlirAttributeIsAType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsATypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirFlatSymbolRefAttrGetValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirFlatSymbolRefAttrGetValue(arg0);
}

export fn fizz_nif_mlirFlatSymbolRefAttrGetValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFlatSymbolRefAttrGetValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirFlatSymbolRefAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirFlatSymbolRefAttrGet(arg0, arg1);
}

export fn fizz_nif_mlirFlatSymbolRefAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFlatSymbolRefAttrGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAFlatSymbolRefWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAFlatSymbolRef(arg0);
}

export fn fizz_nif_mlirAttributeIsAFlatSymbolRef(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAFlatSymbolRefWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolRefAttrGetNestedReferenceWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirSymbolRefAttrGetNestedReference(arg0, arg1);
}

export fn fizz_nif_mlirSymbolRefAttrGetNestedReference(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolRefAttrGetNestedReferenceWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolRefAttrGetNumNestedReferencesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSymbolRefAttrGetNumNestedReferences(arg0);
}

export fn fizz_nif_mlirSymbolRefAttrGetNumNestedReferences(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolRefAttrGetNumNestedReferencesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolRefAttrGetLeafReferenceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSymbolRefAttrGetLeafReference(arg0);
}

export fn fizz_nif_mlirSymbolRefAttrGetLeafReference(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolRefAttrGetLeafReferenceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolRefAttrGetRootReferenceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSymbolRefAttrGetRootReference(arg0);
}

export fn fizz_nif_mlirSymbolRefAttrGetRootReference(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolRefAttrGetRootReferenceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolRefAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirSymbolRefAttrGet(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirSymbolRefAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);
  var arg3: [*c]const c.struct_MlirAttribute = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type__c_ptr_const_c_struct_MlirAttribute, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolRefAttrGetWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsASymbolRefWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsASymbolRef(arg0);
}

export fn fizz_nif_mlirAttributeIsASymbolRef(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsASymbolRefWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirStringAttrGetValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirStringAttrGetValue(arg0);
}

export fn fizz_nif_mlirStringAttrGetValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirStringAttrGetValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirStringAttrTypedGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirStringAttrTypedGet(arg0, arg1);
}

export fn fizz_nif_mlirStringAttrTypedGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirStringAttrTypedGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirStringAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirStringAttrGet(arg0, arg1);
}

export fn fizz_nif_mlirStringAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirStringAttrGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAStringWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAString(arg0);
}

export fn fizz_nif_mlirAttributeIsAString(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAStringWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpaqueAttrGetDataWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpaqueAttrGetData(arg0);
}

export fn fizz_nif_mlirOpaqueAttrGetData(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpaqueAttrGetDataWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpaqueAttrGetDialectNamespaceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpaqueAttrGetDialectNamespace(arg0);
}

export fn fizz_nif_mlirOpaqueAttrGetDialectNamespace(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpaqueAttrGetDialectNamespaceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpaqueAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirOpaqueAttrGet(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirOpaqueAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);
  var arg3: [*c]const u8 = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type__c_ptr_const_u8, args[3]);
  var arg4: c.struct_MlirType = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_c_struct_MlirType, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpaqueAttrGetWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAOpaqueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAOpaque(arg0);
}

export fn fizz_nif_mlirAttributeIsAOpaque(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAOpaqueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAIntegerSetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAIntegerSet(arg0);
}

export fn fizz_nif_mlirAttributeIsAIntegerSet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAIntegerSetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBoolAttrGetValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBoolAttrGetValue(arg0);
}

export fn fizz_nif_mlirBoolAttrGetValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBoolAttrGetValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBoolAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirBoolAttrGet(arg0, arg1);
}

export fn fizz_nif_mlirBoolAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c_int = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_int, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBoolAttrGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsABoolWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsABool(arg0);
}

export fn fizz_nif_mlirAttributeIsABool(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsABoolWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerAttrGetValueUIntWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerAttrGetValueUInt(arg0);
}

export fn fizz_nif_mlirIntegerAttrGetValueUInt(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_u64, @sizeOf(u64));

  const RType = u64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerAttrGetValueUIntWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerAttrGetValueSIntWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerAttrGetValueSInt(arg0);
}

export fn fizz_nif_mlirIntegerAttrGetValueSInt(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerAttrGetValueSIntWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerAttrGetValueIntWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIntegerAttrGetValueInt(arg0);
}

export fn fizz_nif_mlirIntegerAttrGetValueInt(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerAttrGetValueIntWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIntegerAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIntegerAttrGet(arg0, arg1);
}

export fn fizz_nif_mlirIntegerAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: i64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_i64, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIntegerAttrGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAIntegerWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAInteger(arg0);
}

export fn fizz_nif_mlirAttributeIsAInteger(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAIntegerWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirFloatAttrGetValueDoubleWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirFloatAttrGetValueDouble(arg0);
}

export fn fizz_nif_mlirFloatAttrGetValueDouble(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_f64, @sizeOf(f64));

  const RType = f64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFloatAttrGetValueDoubleWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirFloatAttrDoubleGetCheckedWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirFloatAttrDoubleGetChecked(arg0, arg1, arg2);
}

export fn fizz_nif_mlirFloatAttrDoubleGetChecked(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: f64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_f64, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFloatAttrDoubleGetCheckedWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirFloatAttrDoubleGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirFloatAttrDoubleGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirFloatAttrDoubleGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: f64 = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_f64, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirFloatAttrDoubleGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAFloatWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAFloat(arg0);
}

export fn fizz_nif_mlirAttributeIsAFloat(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAFloatWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDictionaryAttrGetElementByNameWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDictionaryAttrGetElementByName(arg0, arg1);
}

export fn fizz_nif_mlirDictionaryAttrGetElementByName(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDictionaryAttrGetElementByNameWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDictionaryAttrGetElementWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDictionaryAttrGetElement(arg0, arg1);
}

export fn fizz_nif_mlirDictionaryAttrGetElement(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirNamedAttribute, @sizeOf(c.struct_MlirNamedAttribute));

  const RType = c.struct_MlirNamedAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDictionaryAttrGetElementWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDictionaryAttrGetNumElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDictionaryAttrGetNumElements(arg0);
}

export fn fizz_nif_mlirDictionaryAttrGetNumElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDictionaryAttrGetNumElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDictionaryAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirDictionaryAttrGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirDictionaryAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirNamedAttribute = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirNamedAttribute, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDictionaryAttrGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsADictionaryWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsADictionary(arg0);
}

export fn fizz_nif_mlirAttributeIsADictionary(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsADictionaryWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirArrayAttrGetElementWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirArrayAttrGetElement(arg0, arg1);
}

export fn fizz_nif_mlirArrayAttrGetElement(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirArrayAttrGetElementWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirArrayAttrGetNumElementsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirArrayAttrGetNumElements(arg0);
}

export fn fizz_nif_mlirArrayAttrGetNumElements(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirArrayAttrGetNumElementsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirArrayAttrGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirArrayAttrGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirArrayAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirAttribute = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirAttribute, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirArrayAttrGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAArrayWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAArray(arg0);
}

export fn fizz_nif_mlirAttributeIsAArray(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAArrayWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapAttrGetValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapAttrGetValue(arg0);
}

export fn fizz_nif_mlirAffineMapAttrGetValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapAttrGetValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapAttrGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapAttrGet(arg0);
}

export fn fizz_nif_mlirAffineMapAttrGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapAttrGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeIsAAffineMapWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeIsAAffineMap(arg0);
}

export fn fizz_nif_mlirAttributeIsAAffineMap(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeIsAAffineMapWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeGetNullWrapper(ret: anytype, ) void {
  ret.* = c.mlirAttributeGetNull();
}

export fn fizz_nif_mlirAttributeGetNull(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeGetNullWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirExternalPassSignalFailureWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirExternalPassSignalFailure(arg0);
}

export fn fizz_nif_mlirExternalPassSignalFailure(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirExternalPass = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirExternalPass, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirExternalPassSignalFailureWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirCreateExternalPassWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype, arg5: anytype, arg6: anytype, arg7: anytype, arg8: anytype) void {
  ret.* = c.mlirCreateExternalPass(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
}

export fn fizz_nif_mlirCreateExternalPass(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirTypeID = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirTypeID, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: c.struct_MlirStringRef = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirStringRef, args[2]);
  var arg3: c.struct_MlirStringRef = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_struct_MlirStringRef, args[3]);
  var arg4: c.struct_MlirStringRef = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_c_struct_MlirStringRef, args[4]);
  var arg5: isize = undefined; arg5 = beam.fetch_resource(arg5, env, resource_type_isize, args[5]);
  var arg6: [*c]c.struct_MlirDialectHandle = undefined; arg6 = beam.fetch_resource(arg6, env, resource_type__c_ptr_c_struct_MlirDialectHandle, args[6]);
  var arg7: c.struct_MlirExternalPassCallbacks = undefined; arg7 = beam.fetch_resource(arg7, env, resource_type_c_struct_MlirExternalPassCallbacks, args[7]);
  var arg8: ?*anyopaque = undefined; arg8 = beam.fetch_resource(arg8, env, resource_type_void_ptr, args[8]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPass, @sizeOf(c.struct_MlirPass));

  const RType = c.struct_MlirPass;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirCreateExternalPassWrapper(obj, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
  return e.enif_make_resource(env, ptr);
}

fn mlirParsePassPipelineWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirParsePassPipeline(arg0, arg1);
}

export fn fizz_nif_mlirParsePassPipeline(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPassManager, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLogicalResult, @sizeOf(c.struct_MlirLogicalResult));

  const RType = c.struct_MlirLogicalResult;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirParsePassPipelineWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirPrintPassPipelineWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirPrintPassPipeline(arg0, arg1, arg2);
}

export fn fizz_nif_mlirPrintPassPipeline(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPassManager, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPrintPassPipelineWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpPassManagerAddOwnedPassWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOpPassManagerAddOwnedPass(arg0, arg1);
}

export fn fizz_nif_mlirOpPassManagerAddOwnedPass(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPassManager, args[0]);
  var arg1: c.struct_MlirPass = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirPass, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpPassManagerAddOwnedPassWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirPassManagerAddOwnedPassWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirPassManagerAddOwnedPass(arg0, arg1);
}

export fn fizz_nif_mlirPassManagerAddOwnedPass(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirPassManager, args[0]);
  var arg1: c.struct_MlirPass = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirPass, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPassManagerAddOwnedPassWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpPassManagerGetNestedUnderWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOpPassManagerGetNestedUnder(arg0, arg1);
}

export fn fizz_nif_mlirOpPassManagerGetNestedUnder(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPassManager, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOpPassManager, @sizeOf(c.struct_MlirOpPassManager));

  const RType = c.struct_MlirOpPassManager;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpPassManagerGetNestedUnderWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirPassManagerGetNestedUnderWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirPassManagerGetNestedUnder(arg0, arg1);
}

export fn fizz_nif_mlirPassManagerGetNestedUnder(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirPassManager, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOpPassManager, @sizeOf(c.struct_MlirOpPassManager));

  const RType = c.struct_MlirOpPassManager;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPassManagerGetNestedUnderWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirPassManagerEnableVerifierWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirPassManagerEnableVerifier(arg0, arg1);
}

export fn fizz_nif_mlirPassManagerEnableVerifier(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirPassManager, args[0]);
  var arg1: bool = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_bool, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPassManagerEnableVerifierWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirPassManagerEnableIRPrintingWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPassManagerEnableIRPrinting(arg0);
}

export fn fizz_nif_mlirPassManagerEnableIRPrinting(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirPassManager, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPassManagerEnableIRPrintingWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirPassManagerRunWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirPassManagerRun(arg0, arg1);
}

export fn fizz_nif_mlirPassManagerRun(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirPassManager, args[0]);
  var arg1: c.struct_MlirModule = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirModule, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLogicalResult, @sizeOf(c.struct_MlirLogicalResult));

  const RType = c.struct_MlirLogicalResult;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPassManagerRunWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirPassManagerGetAsOpPassManagerWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPassManagerGetAsOpPassManager(arg0);
}

export fn fizz_nif_mlirPassManagerGetAsOpPassManager(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirPassManager, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOpPassManager, @sizeOf(c.struct_MlirOpPassManager));

  const RType = c.struct_MlirOpPassManager;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPassManagerGetAsOpPassManagerWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirPassManagerDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPassManagerDestroy(arg0);
}

export fn fizz_nif_mlirPassManagerDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirPassManager = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirPassManager, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPassManagerDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirPassManagerCreateWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirPassManagerCreate(arg0);
}

export fn fizz_nif_mlirPassManagerCreate(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirPassManager, @sizeOf(c.struct_MlirPassManager));

  const RType = c.struct_MlirPassManager;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirPassManagerCreateWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAllPassesWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegisterAllPasses();
}

export fn fizz_nif_mlirRegisterAllPasses(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAllPassesWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAllLLVMTranslationsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirRegisterAllLLVMTranslations(arg0);
}

export fn fizz_nif_mlirRegisterAllLLVMTranslations(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAllLLVMTranslationsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegisterAllDialectsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirRegisterAllDialects(arg0);
}

export fn fizz_nif_mlirRegisterAllDialects(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegisterAllDialectsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectHandleLoadDialectWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDialectHandleLoadDialect(arg0, arg1);
}

export fn fizz_nif_mlirDialectHandleLoadDialect(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDialectHandle = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDialectHandle, args[0]);
  var arg1: c.struct_MlirContext = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirContext, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialect, @sizeOf(c.struct_MlirDialect));

  const RType = c.struct_MlirDialect;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectHandleLoadDialectWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectHandleRegisterDialectWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDialectHandleRegisterDialect(arg0, arg1);
}

export fn fizz_nif_mlirDialectHandleRegisterDialect(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDialectHandle = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDialectHandle, args[0]);
  var arg1: c.struct_MlirContext = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirContext, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectHandleRegisterDialectWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectHandleInsertDialectWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDialectHandleInsertDialect(arg0, arg1);
}

export fn fizz_nif_mlirDialectHandleInsertDialect(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDialectHandle = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDialectHandle, args[0]);
  var arg1: c.struct_MlirDialectRegistry = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirDialectRegistry, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectHandleInsertDialectWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectHandleGetNamespaceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDialectHandleGetNamespace(arg0);
}

export fn fizz_nif_mlirDialectHandleGetNamespace(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDialectHandle = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDialectHandle, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectHandleGetNamespaceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapCompressUnusedSymbolsWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirAffineMapCompressUnusedSymbols(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirAffineMapCompressUnusedSymbols(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_c_struct_MlirAffineMap, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);
  var arg3: ?fn(?*anyopaque, isize, c.struct_MlirAffineMap) callconv(.C) void = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type__nullable_fn__nullable__pointer_anyopaque___isize___c_struct_MlirAffineMap__callconv__C__void, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapCompressUnusedSymbolsWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapReplaceWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirAffineMapReplace(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirAffineMapReplace(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);
  var arg1: c.struct_MlirAffineExpr = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineExpr, args[1]);
  var arg2: c.struct_MlirAffineExpr = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirAffineExpr, args[2]);
  var arg3: isize = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_isize, args[3]);
  var arg4: isize = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type_isize, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapReplaceWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetMinorSubMapWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineMapGetMinorSubMap(arg0, arg1);
}

export fn fizz_nif_mlirAffineMapGetMinorSubMap(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetMinorSubMapWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetMajorSubMapWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineMapGetMajorSubMap(arg0, arg1);
}

export fn fizz_nif_mlirAffineMapGetMajorSubMap(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetMajorSubMapWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetSubMapWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirAffineMapGetSubMap(arg0, arg1, arg2);
}

export fn fizz_nif_mlirAffineMapGetSubMap(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_isize, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetSubMapWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapIsPermutationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapIsPermutation(arg0);
}

export fn fizz_nif_mlirAffineMapIsPermutation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapIsPermutationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapIsProjectedPermutationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapIsProjectedPermutation(arg0);
}

export fn fizz_nif_mlirAffineMapIsProjectedPermutation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapIsProjectedPermutationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetNumInputsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapGetNumInputs(arg0);
}

export fn fizz_nif_mlirAffineMapGetNumInputs(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetNumInputsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetResultWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineMapGetResult(arg0, arg1);
}

export fn fizz_nif_mlirAffineMapGetResult(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetResultWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetNumResultsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapGetNumResults(arg0);
}

export fn fizz_nif_mlirAffineMapGetNumResults(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetNumResultsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetNumSymbolsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapGetNumSymbols(arg0);
}

export fn fizz_nif_mlirAffineMapGetNumSymbols(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetNumSymbolsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetNumDimsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapGetNumDims(arg0);
}

export fn fizz_nif_mlirAffineMapGetNumDims(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetNumDimsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetSingleConstantResultWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapGetSingleConstantResult(arg0);
}

export fn fizz_nif_mlirAffineMapGetSingleConstantResult(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetSingleConstantResultWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapIsSingleConstantWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapIsSingleConstant(arg0);
}

export fn fizz_nif_mlirAffineMapIsSingleConstant(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapIsSingleConstantWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapIsEmptyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapIsEmpty(arg0);
}

export fn fizz_nif_mlirAffineMapIsEmpty(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapIsEmptyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapIsMinorIdentityWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapIsMinorIdentity(arg0);
}

export fn fizz_nif_mlirAffineMapIsMinorIdentity(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapIsMinorIdentityWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapIsIdentityWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapIsIdentity(arg0);
}

export fn fizz_nif_mlirAffineMapIsIdentity(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapIsIdentityWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapPermutationGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirAffineMapPermutationGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirAffineMapPermutationGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]c_uint = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_c_uint, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapPermutationGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapMinorIdentityGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirAffineMapMinorIdentityGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirAffineMapMinorIdentityGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapMinorIdentityGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapMultiDimIdentityGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineMapMultiDimIdentityGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineMapMultiDimIdentityGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapMultiDimIdentityGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapConstantGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineMapConstantGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineMapConstantGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: i64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_i64, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapConstantGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype, arg4: anytype) void {
  ret.* = c.mlirAffineMapGet(arg0, arg1, arg2, arg3, arg4);
}

export fn fizz_nif_mlirAffineMapGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);
  var arg3: isize = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_isize, args[3]);
  var arg4: [*c]c.struct_MlirAffineExpr = undefined; arg4 = beam.fetch_resource(arg4, env, resource_type__c_ptr_c_struct_MlirAffineExpr, args[4]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetWrapper(obj, arg0, arg1, arg2, arg3, arg4);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapZeroResultGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirAffineMapZeroResultGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirAffineMapZeroResultGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: isize = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_isize, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapZeroResultGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapEmptyGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapEmptyGet(arg0);
}

export fn fizz_nif_mlirAffineMapEmptyGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineMap, @sizeOf(c.struct_MlirAffineMap));

  const RType = c.struct_MlirAffineMap;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapEmptyGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapDumpWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapDump(arg0);
}

export fn fizz_nif_mlirAffineMapDump(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapDumpWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapPrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirAffineMapPrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirAffineMapPrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapPrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineMapEqual(arg0, arg1);
}

export fn fizz_nif_mlirAffineMapEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);
  var arg1: c.struct_MlirAffineMap = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineMap, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMapGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineMapGetContext(arg0);
}

export fn fizz_nif_mlirAffineMapGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineMap = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineMap, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMapGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineBinaryOpExprGetRHSWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineBinaryOpExprGetRHS(arg0);
}

export fn fizz_nif_mlirAffineBinaryOpExprGetRHS(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineBinaryOpExprGetRHSWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineBinaryOpExprGetLHSWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineBinaryOpExprGetLHS(arg0);
}

export fn fizz_nif_mlirAffineBinaryOpExprGetLHS(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineBinaryOpExprGetLHSWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsABinaryWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsABinary(arg0);
}

export fn fizz_nif_mlirAffineExprIsABinary(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsABinaryWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineCeilDivExprGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineCeilDivExprGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineCeilDivExprGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: c.struct_MlirAffineExpr = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineExpr, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineCeilDivExprGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsACeilDivWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsACeilDiv(arg0);
}

export fn fizz_nif_mlirAffineExprIsACeilDiv(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsACeilDivWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineFloorDivExprGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineFloorDivExprGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineFloorDivExprGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: c.struct_MlirAffineExpr = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineExpr, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineFloorDivExprGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsAFloorDivWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsAFloorDiv(arg0);
}

export fn fizz_nif_mlirAffineExprIsAFloorDiv(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsAFloorDivWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineModExprGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineModExprGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineModExprGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: c.struct_MlirAffineExpr = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineExpr, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineModExprGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsAModWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsAMod(arg0);
}

export fn fizz_nif_mlirAffineExprIsAMod(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsAModWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineMulExprGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineMulExprGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineMulExprGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: c.struct_MlirAffineExpr = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineExpr, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineMulExprGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsAMulWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsAMul(arg0);
}

export fn fizz_nif_mlirAffineExprIsAMul(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsAMulWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineAddExprGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineAddExprGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineAddExprGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: c.struct_MlirAffineExpr = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineExpr, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineAddExprGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsAAddWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsAAdd(arg0);
}

export fn fizz_nif_mlirAffineExprIsAAdd(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsAAddWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineConstantExprGetValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineConstantExprGetValue(arg0);
}

export fn fizz_nif_mlirAffineConstantExprGetValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineConstantExprGetValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineConstantExprGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineConstantExprGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineConstantExprGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: i64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_i64, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineConstantExprGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsAConstantWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsAConstant(arg0);
}

export fn fizz_nif_mlirAffineExprIsAConstant(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsAConstantWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineSymbolExprGetPositionWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineSymbolExprGetPosition(arg0);
}

export fn fizz_nif_mlirAffineSymbolExprGetPosition(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineSymbolExprGetPositionWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineSymbolExprGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineSymbolExprGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineSymbolExprGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineSymbolExprGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsASymbolWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsASymbol(arg0);
}

export fn fizz_nif_mlirAffineExprIsASymbol(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsASymbolWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineDimExprGetPositionWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineDimExprGetPosition(arg0);
}

export fn fizz_nif_mlirAffineDimExprGetPosition(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineDimExprGetPositionWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineDimExprGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineDimExprGet(arg0, arg1);
}

export fn fizz_nif_mlirAffineDimExprGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineDimExprGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsADimWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsADim(arg0);
}

export fn fizz_nif_mlirAffineExprIsADim(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsADimWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprComposeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineExprCompose(arg0, arg1);
}

export fn fizz_nif_mlirAffineExprCompose(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: c.struct_MlirAffineMap = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineMap, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAffineExpr, @sizeOf(c.struct_MlirAffineExpr));

  const RType = c.struct_MlirAffineExpr;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprComposeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsFunctionOfDimWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineExprIsFunctionOfDim(arg0, arg1);
}

export fn fizz_nif_mlirAffineExprIsFunctionOfDim(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsFunctionOfDimWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsMultipleOfWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineExprIsMultipleOf(arg0, arg1);
}

export fn fizz_nif_mlirAffineExprIsMultipleOf(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: i64 = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_i64, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsMultipleOfWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprGetLargestKnownDivisorWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprGetLargestKnownDivisor(arg0);
}

export fn fizz_nif_mlirAffineExprGetLargestKnownDivisor(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_i64, @sizeOf(i64));

  const RType = i64;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprGetLargestKnownDivisorWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsPureAffineWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsPureAffine(arg0);
}

export fn fizz_nif_mlirAffineExprIsPureAffine(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsPureAffineWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprIsSymbolicOrConstantWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprIsSymbolicOrConstant(arg0);
}

export fn fizz_nif_mlirAffineExprIsSymbolicOrConstant(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprIsSymbolicOrConstantWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprDumpWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprDump(arg0);
}

export fn fizz_nif_mlirAffineExprDump(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprDumpWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprPrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirAffineExprPrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirAffineExprPrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprPrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAffineExprEqual(arg0, arg1);
}

export fn fizz_nif_mlirAffineExprEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);
  var arg1: c.struct_MlirAffineExpr = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAffineExpr, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAffineExprGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAffineExprGetContext(arg0);
}

export fn fizz_nif_mlirAffineExprGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAffineExpr = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAffineExpr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAffineExprGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableWalkSymbolTablesWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirSymbolTableWalkSymbolTables(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirSymbolTableWalkSymbolTables(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: bool = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_bool, args[1]);
  var arg2: ?fn(c.struct_MlirOperation, bool, ?*anyopaque) callconv(.C) void = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__nullable_fn_c_struct_MlirOperation___bool____nullable__pointer_anyopaque__callconv__C__void, args[2]);
  var arg3: ?*anyopaque = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_void_ptr, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableWalkSymbolTablesWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableReplaceAllSymbolUsesWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirSymbolTableReplaceAllSymbolUses(arg0, arg1, arg2);
}

export fn fizz_nif_mlirSymbolTableReplaceAllSymbolUses(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirStringRef = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirStringRef, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: c.struct_MlirOperation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirOperation, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLogicalResult, @sizeOf(c.struct_MlirLogicalResult));

  const RType = c.struct_MlirLogicalResult;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableReplaceAllSymbolUsesWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableEraseWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirSymbolTableErase(arg0, arg1);
}

export fn fizz_nif_mlirSymbolTableErase(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirSymbolTable = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirSymbolTable, args[0]);
  var arg1: c.struct_MlirOperation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOperation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableEraseWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableInsertWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirSymbolTableInsert(arg0, arg1);
}

export fn fizz_nif_mlirSymbolTableInsert(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirSymbolTable = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirSymbolTable, args[0]);
  var arg1: c.struct_MlirOperation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOperation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableInsertWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableLookupWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirSymbolTableLookup(arg0, arg1);
}

export fn fizz_nif_mlirSymbolTableLookup(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirSymbolTable = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirSymbolTable, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableLookupWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSymbolTableDestroy(arg0);
}

export fn fizz_nif_mlirSymbolTableDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirSymbolTable = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirSymbolTable, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableCreateWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirSymbolTableCreate(arg0);
}

export fn fizz_nif_mlirSymbolTableCreate(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirSymbolTable, @sizeOf(c.struct_MlirSymbolTable));

  const RType = c.struct_MlirSymbolTable;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableCreateWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableGetVisibilityAttributeNameWrapper(ret: anytype, ) void {
  ret.* = c.mlirSymbolTableGetVisibilityAttributeName();
}

export fn fizz_nif_mlirSymbolTableGetVisibilityAttributeName(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableGetVisibilityAttributeNameWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirSymbolTableGetSymbolAttributeNameWrapper(ret: anytype, ) void {
  ret.* = c.mlirSymbolTableGetSymbolAttributeName();
}

export fn fizz_nif_mlirSymbolTableGetSymbolAttributeName(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirSymbolTableGetSymbolAttributeNameWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirIdentifierStrWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIdentifierStr(arg0);
}

export fn fizz_nif_mlirIdentifierStr(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIdentifier = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIdentifier, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIdentifierStrWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIdentifierEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIdentifierEqual(arg0, arg1);
}

export fn fizz_nif_mlirIdentifierEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIdentifier = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIdentifier, args[0]);
  var arg1: c.struct_MlirIdentifier = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirIdentifier, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIdentifierEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirIdentifierGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirIdentifierGetContext(arg0);
}

export fn fizz_nif_mlirIdentifierGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIdentifier = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIdentifier, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIdentifierGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirIdentifierGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirIdentifierGet(arg0, arg1);
}

export fn fizz_nif_mlirIdentifierGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirIdentifier, @sizeOf(c.struct_MlirIdentifier));

  const RType = c.struct_MlirIdentifier;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirIdentifierGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirNamedAttributeGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirNamedAttributeGet(arg0, arg1);
}

export fn fizz_nif_mlirNamedAttributeGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirIdentifier = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirIdentifier, args[0]);
  var arg1: c.struct_MlirAttribute = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAttribute, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirNamedAttribute, @sizeOf(c.struct_MlirNamedAttribute));

  const RType = c.struct_MlirNamedAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirNamedAttributeGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeDumpWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeDump(arg0);
}

export fn fizz_nif_mlirAttributeDump(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeDumpWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributePrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirAttributePrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirAttributePrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributePrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAttributeEqual(arg0, arg1);
}

export fn fizz_nif_mlirAttributeEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);
  var arg1: c.struct_MlirAttribute = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirAttribute, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeGetTypeIDWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeGetTypeID(arg0);
}

export fn fizz_nif_mlirAttributeGetTypeID(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirTypeID, @sizeOf(c.struct_MlirTypeID));

  const RType = c.struct_MlirTypeID;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeGetTypeIDWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeGetTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeGetType(arg0);
}

export fn fizz_nif_mlirAttributeGetType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeGetTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirAttributeGetContext(arg0);
}

export fn fizz_nif_mlirAttributeGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirAttribute = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirAttribute, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirAttributeParseGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirAttributeParseGet(arg0, arg1);
}

export fn fizz_nif_mlirAttributeParseGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirAttributeParseGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeDumpWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeDump(arg0);
}

export fn fizz_nif_mlirTypeDump(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeDumpWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypePrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirTypePrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirTypePrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypePrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirTypeEqual(arg0, arg1);
}

export fn fizz_nif_mlirTypeEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeGetTypeIDWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeGetTypeID(arg0);
}

export fn fizz_nif_mlirTypeGetTypeID(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirTypeID, @sizeOf(c.struct_MlirTypeID));

  const RType = c.struct_MlirTypeID;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeGetTypeIDWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeGetContext(arg0);
}

export fn fizz_nif_mlirTypeGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirType = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirType, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeParseGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirTypeParseGet(arg0, arg1);
}

export fn fizz_nif_mlirTypeParseGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeParseGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirValuePrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirValuePrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirValuePrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirValuePrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirValueDumpWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirValueDump(arg0);
}

export fn fizz_nif_mlirValueDump(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirValueDumpWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirValueGetTypeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirValueGetType(arg0);
}

export fn fizz_nif_mlirValueGetType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirType, @sizeOf(c.struct_MlirType));

  const RType = c.struct_MlirType;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirValueGetTypeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpResultGetResultNumberWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpResultGetResultNumber(arg0);
}

export fn fizz_nif_mlirOpResultGetResultNumber(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpResultGetResultNumberWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpResultGetOwnerWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpResultGetOwner(arg0);
}

export fn fizz_nif_mlirOpResultGetOwner(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpResultGetOwnerWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockArgumentSetTypeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirBlockArgumentSetType(arg0, arg1);
}

export fn fizz_nif_mlirBlockArgumentSetType(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockArgumentSetTypeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockArgumentGetArgNumberWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockArgumentGetArgNumber(arg0);
}

export fn fizz_nif_mlirBlockArgumentGetArgNumber(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockArgumentGetArgNumberWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockArgumentGetOwnerWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockArgumentGetOwner(arg0);
}

export fn fizz_nif_mlirBlockArgumentGetOwner(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirBlock, @sizeOf(c.struct_MlirBlock));

  const RType = c.struct_MlirBlock;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockArgumentGetOwnerWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirValueIsAOpResultWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirValueIsAOpResult(arg0);
}

export fn fizz_nif_mlirValueIsAOpResult(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirValueIsAOpResultWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirValueIsABlockArgumentWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirValueIsABlockArgument(arg0);
}

export fn fizz_nif_mlirValueIsABlockArgument(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirValueIsABlockArgumentWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirValueEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirValueEqual(arg0, arg1);
}

export fn fizz_nif_mlirValueEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirValue = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirValue, args[0]);
  var arg1: c.struct_MlirValue = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirValue, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirValueEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockPrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirBlockPrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirBlockPrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockPrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockGetArgumentWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirBlockGetArgument(arg0, arg1);
}

export fn fizz_nif_mlirBlockGetArgument(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirValue, @sizeOf(c.struct_MlirValue));

  const RType = c.struct_MlirValue;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockGetArgumentWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockAddArgumentWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirBlockAddArgument(arg0, arg1, arg2);
}

export fn fizz_nif_mlirBlockAddArgument(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);
  var arg1: c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirType, args[1]);
  var arg2: c.struct_MlirLocation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirLocation, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirValue, @sizeOf(c.struct_MlirValue));

  const RType = c.struct_MlirValue;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockAddArgumentWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockGetNumArgumentsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockGetNumArguments(arg0);
}

export fn fizz_nif_mlirBlockGetNumArguments(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockGetNumArgumentsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockInsertOwnedOperationBeforeWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirBlockInsertOwnedOperationBefore(arg0, arg1, arg2);
}

export fn fizz_nif_mlirBlockInsertOwnedOperationBefore(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);
  var arg1: c.struct_MlirOperation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOperation, args[1]);
  var arg2: c.struct_MlirOperation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirOperation, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockInsertOwnedOperationBeforeWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockInsertOwnedOperationAfterWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirBlockInsertOwnedOperationAfter(arg0, arg1, arg2);
}

export fn fizz_nif_mlirBlockInsertOwnedOperationAfter(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);
  var arg1: c.struct_MlirOperation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOperation, args[1]);
  var arg2: c.struct_MlirOperation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirOperation, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockInsertOwnedOperationAfterWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockInsertOwnedOperationWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirBlockInsertOwnedOperation(arg0, arg1, arg2);
}

export fn fizz_nif_mlirBlockInsertOwnedOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: c.struct_MlirOperation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirOperation, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockInsertOwnedOperationWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockAppendOwnedOperationWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirBlockAppendOwnedOperation(arg0, arg1);
}

export fn fizz_nif_mlirBlockAppendOwnedOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);
  var arg1: c.struct_MlirOperation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOperation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockAppendOwnedOperationWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockGetTerminatorWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockGetTerminator(arg0);
}

export fn fizz_nif_mlirBlockGetTerminator(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockGetTerminatorWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockGetFirstOperationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockGetFirstOperation(arg0);
}

export fn fizz_nif_mlirBlockGetFirstOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockGetFirstOperationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockGetNextInRegionWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockGetNextInRegion(arg0);
}

export fn fizz_nif_mlirBlockGetNextInRegion(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirBlock, @sizeOf(c.struct_MlirBlock));

  const RType = c.struct_MlirBlock;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockGetNextInRegionWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockGetParentRegionWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockGetParentRegion(arg0);
}

export fn fizz_nif_mlirBlockGetParentRegion(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirRegion, @sizeOf(c.struct_MlirRegion));

  const RType = c.struct_MlirRegion;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockGetParentRegionWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockGetParentOperationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockGetParentOperation(arg0);
}

export fn fizz_nif_mlirBlockGetParentOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockGetParentOperationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirBlockEqual(arg0, arg1);
}

export fn fizz_nif_mlirBlockEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);
  var arg1: c.struct_MlirBlock = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirBlock, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockDetachWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockDetach(arg0);
}

export fn fizz_nif_mlirBlockDetach(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockDetachWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirBlockDestroy(arg0);
}

export fn fizz_nif_mlirBlockDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirBlock = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirBlock, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirBlockCreateWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirBlockCreate(arg0, arg1, arg2);
}

export fn fizz_nif_mlirBlockCreate(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: isize = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_isize, args[0]);
  var arg1: [*c]const c.struct_MlirType = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__c_ptr_const_c_struct_MlirType, args[1]);
  var arg2: [*c]const c.struct_MlirLocation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirLocation, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirBlock, @sizeOf(c.struct_MlirBlock));

  const RType = c.struct_MlirBlock;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirBlockCreateWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionGetNextInOperationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirRegionGetNextInOperation(arg0);
}

export fn fizz_nif_mlirRegionGetNextInOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirRegion = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirRegion, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirRegion, @sizeOf(c.struct_MlirRegion));

  const RType = c.struct_MlirRegion;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionGetNextInOperationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetFirstRegionWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetFirstRegion(arg0);
}

export fn fizz_nif_mlirOperationGetFirstRegion(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirRegion, @sizeOf(c.struct_MlirRegion));

  const RType = c.struct_MlirRegion;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetFirstRegionWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionInsertOwnedBlockBeforeWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirRegionInsertOwnedBlockBefore(arg0, arg1, arg2);
}

export fn fizz_nif_mlirRegionInsertOwnedBlockBefore(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirRegion = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirRegion, args[0]);
  var arg1: c.struct_MlirBlock = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirBlock, args[1]);
  var arg2: c.struct_MlirBlock = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirBlock, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionInsertOwnedBlockBeforeWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionInsertOwnedBlockAfterWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirRegionInsertOwnedBlockAfter(arg0, arg1, arg2);
}

export fn fizz_nif_mlirRegionInsertOwnedBlockAfter(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirRegion = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirRegion, args[0]);
  var arg1: c.struct_MlirBlock = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirBlock, args[1]);
  var arg2: c.struct_MlirBlock = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirBlock, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionInsertOwnedBlockAfterWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionInsertOwnedBlockWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirRegionInsertOwnedBlock(arg0, arg1, arg2);
}

export fn fizz_nif_mlirRegionInsertOwnedBlock(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirRegion = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirRegion, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: c.struct_MlirBlock = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirBlock, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionInsertOwnedBlockWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionAppendOwnedBlockWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirRegionAppendOwnedBlock(arg0, arg1);
}

export fn fizz_nif_mlirRegionAppendOwnedBlock(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirRegion = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirRegion, args[0]);
  var arg1: c.struct_MlirBlock = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirBlock, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionAppendOwnedBlockWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionGetFirstBlockWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirRegionGetFirstBlock(arg0);
}

export fn fizz_nif_mlirRegionGetFirstBlock(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirRegion = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirRegion, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirBlock, @sizeOf(c.struct_MlirBlock));

  const RType = c.struct_MlirBlock;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionGetFirstBlockWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirRegionEqual(arg0, arg1);
}

export fn fizz_nif_mlirRegionEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirRegion = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirRegion, args[0]);
  var arg1: c.struct_MlirRegion = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirRegion, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirRegionDestroy(arg0);
}

export fn fizz_nif_mlirRegionDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirRegion = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirRegion, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirRegionCreateWrapper(ret: anytype, ) void {
  ret.* = c.mlirRegionCreate();
}

export fn fizz_nif_mlirRegionCreate(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirRegion, @sizeOf(c.struct_MlirRegion));

  const RType = c.struct_MlirRegion;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirRegionCreateWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationMoveBeforeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationMoveBefore(arg0, arg1);
}

export fn fizz_nif_mlirOperationMoveBefore(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: c.struct_MlirOperation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOperation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationMoveBeforeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationMoveAfterWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationMoveAfter(arg0, arg1);
}

export fn fizz_nif_mlirOperationMoveAfter(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: c.struct_MlirOperation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOperation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationMoveAfterWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationVerifyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationVerify(arg0);
}

export fn fizz_nif_mlirOperationVerify(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationVerifyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationDumpWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationDump(arg0);
}

export fn fizz_nif_mlirOperationDump(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationDumpWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationPrintWithFlagsWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirOperationPrintWithFlags(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirOperationPrintWithFlags(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: c.struct_MlirOpPrintingFlags = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOpPrintingFlags, args[1]);
  var arg2: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[2]);
  var arg3: ?*anyopaque = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_void_ptr, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationPrintWithFlagsWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationPrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationPrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationPrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationPrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationRemoveAttributeByNameWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationRemoveAttributeByName(arg0, arg1);
}

export fn fizz_nif_mlirOperationRemoveAttributeByName(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationRemoveAttributeByNameWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationSetAttributeByNameWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationSetAttributeByName(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationSetAttributeByName(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: c.struct_MlirAttribute = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirAttribute, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationSetAttributeByNameWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetAttributeByNameWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationGetAttributeByName(arg0, arg1);
}

export fn fizz_nif_mlirOperationGetAttributeByName(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirAttribute, @sizeOf(c.struct_MlirAttribute));

  const RType = c.struct_MlirAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetAttributeByNameWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetAttributeWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationGetAttribute(arg0, arg1);
}

export fn fizz_nif_mlirOperationGetAttribute(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirNamedAttribute, @sizeOf(c.struct_MlirNamedAttribute));

  const RType = c.struct_MlirNamedAttribute;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetAttributeWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetNumAttributesWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetNumAttributes(arg0);
}

export fn fizz_nif_mlirOperationGetNumAttributes(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetNumAttributesWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetSuccessorWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationGetSuccessor(arg0, arg1);
}

export fn fizz_nif_mlirOperationGetSuccessor(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirBlock, @sizeOf(c.struct_MlirBlock));

  const RType = c.struct_MlirBlock;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetSuccessorWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetNumSuccessorsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetNumSuccessors(arg0);
}

export fn fizz_nif_mlirOperationGetNumSuccessors(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetNumSuccessorsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetResultWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationGetResult(arg0, arg1);
}

export fn fizz_nif_mlirOperationGetResult(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirValue, @sizeOf(c.struct_MlirValue));

  const RType = c.struct_MlirValue;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetResultWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetNumResultsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetNumResults(arg0);
}

export fn fizz_nif_mlirOperationGetNumResults(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetNumResultsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationSetOperandWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationSetOperand(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationSetOperand(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: c.struct_MlirValue = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirValue, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationSetOperandWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetOperandWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationGetOperand(arg0, arg1);
}

export fn fizz_nif_mlirOperationGetOperand(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirValue, @sizeOf(c.struct_MlirValue));

  const RType = c.struct_MlirValue;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetOperandWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetNumOperandsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetNumOperands(arg0);
}

export fn fizz_nif_mlirOperationGetNumOperands(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetNumOperandsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetNextInBlockWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetNextInBlock(arg0);
}

export fn fizz_nif_mlirOperationGetNextInBlock(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetNextInBlockWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetRegionWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationGetRegion(arg0, arg1);
}

export fn fizz_nif_mlirOperationGetRegion(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirRegion, @sizeOf(c.struct_MlirRegion));

  const RType = c.struct_MlirRegion;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetRegionWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetNumRegionsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetNumRegions(arg0);
}

export fn fizz_nif_mlirOperationGetNumRegions(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetNumRegionsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetParentOperationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetParentOperation(arg0);
}

export fn fizz_nif_mlirOperationGetParentOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetParentOperationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetBlockWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetBlock(arg0);
}

export fn fizz_nif_mlirOperationGetBlock(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirBlock, @sizeOf(c.struct_MlirBlock));

  const RType = c.struct_MlirBlock;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetBlockWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetNameWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetName(arg0);
}

export fn fizz_nif_mlirOperationGetName(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirIdentifier, @sizeOf(c.struct_MlirIdentifier));

  const RType = c.struct_MlirIdentifier;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetNameWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetTypeIDWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetTypeID(arg0);
}

export fn fizz_nif_mlirOperationGetTypeID(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirTypeID, @sizeOf(c.struct_MlirTypeID));

  const RType = c.struct_MlirTypeID;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetTypeIDWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetLocationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetLocation(arg0);
}

export fn fizz_nif_mlirOperationGetLocation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLocation, @sizeOf(c.struct_MlirLocation));

  const RType = c.struct_MlirLocation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetLocationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationGetContext(arg0);
}

export fn fizz_nif_mlirOperationGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationEqual(arg0, arg1);
}

export fn fizz_nif_mlirOperationEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);
  var arg1: c.struct_MlirOperation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirOperation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationRemoveFromParentWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationRemoveFromParent(arg0);
}

export fn fizz_nif_mlirOperationRemoveFromParent(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationRemoveFromParentWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationDestroy(arg0);
}

export fn fizz_nif_mlirOperationDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationCloneWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationClone(arg0);
}

export fn fizz_nif_mlirOperationClone(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationCloneWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationCreateWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationCreate(arg0);
}

export fn fizz_nif_mlirOperationCreate(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]c.struct_MlirOperationState = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_c_struct_MlirOperationState, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationCreateWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpPrintingFlagsUseLocalScopeWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpPrintingFlagsUseLocalScope(arg0);
}

export fn fizz_nif_mlirOpPrintingFlagsUseLocalScope(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPrintingFlags = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPrintingFlags, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpPrintingFlagsUseLocalScopeWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpPrintingFlagsPrintGenericOpFormWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpPrintingFlagsPrintGenericOpForm(arg0);
}

export fn fizz_nif_mlirOpPrintingFlagsPrintGenericOpForm(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPrintingFlags = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPrintingFlags, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpPrintingFlagsPrintGenericOpFormWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpPrintingFlagsEnableDebugInfoWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOpPrintingFlagsEnableDebugInfo(arg0, arg1);
}

export fn fizz_nif_mlirOpPrintingFlagsEnableDebugInfo(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPrintingFlags = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPrintingFlags, args[0]);
  var arg1: bool = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_bool, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpPrintingFlagsEnableDebugInfoWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpPrintingFlagsElideLargeElementsAttrsWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOpPrintingFlagsElideLargeElementsAttrs(arg0, arg1);
}

export fn fizz_nif_mlirOpPrintingFlagsElideLargeElementsAttrs(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPrintingFlags = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPrintingFlags, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpPrintingFlagsElideLargeElementsAttrsWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpPrintingFlagsDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOpPrintingFlagsDestroy(arg0);
}

export fn fizz_nif_mlirOpPrintingFlagsDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOpPrintingFlags = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOpPrintingFlags, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpPrintingFlagsDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOpPrintingFlagsCreateWrapper(ret: anytype, ) void {
  ret.* = c.mlirOpPrintingFlagsCreate();
}

export fn fizz_nif_mlirOpPrintingFlagsCreate(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOpPrintingFlags, @sizeOf(c.struct_MlirOpPrintingFlags));

  const RType = c.struct_MlirOpPrintingFlags;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOpPrintingFlagsCreateWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationStateEnableResultTypeInferenceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirOperationStateEnableResultTypeInference(arg0);
}

export fn fizz_nif_mlirOperationStateEnableResultTypeInference(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]c.struct_MlirOperationState = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_c_struct_MlirOperationState, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationStateEnableResultTypeInferenceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationStateAddAttributesWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationStateAddAttributes(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationStateAddAttributes(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]c.struct_MlirOperationState = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_c_struct_MlirOperationState, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirNamedAttribute = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirNamedAttribute, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationStateAddAttributesWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationStateAddSuccessorsWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationStateAddSuccessors(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationStateAddSuccessors(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]c.struct_MlirOperationState = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_c_struct_MlirOperationState, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirBlock = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirBlock, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationStateAddSuccessorsWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationStateAddOwnedRegionsWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationStateAddOwnedRegions(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationStateAddOwnedRegions(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]c.struct_MlirOperationState = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_c_struct_MlirOperationState, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirRegion = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirRegion, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationStateAddOwnedRegionsWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationStateAddOperandsWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationStateAddOperands(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationStateAddOperands(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]c.struct_MlirOperationState = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_c_struct_MlirOperationState, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirValue = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirValue, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationStateAddOperandsWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationStateAddResultsWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirOperationStateAddResults(arg0, arg1, arg2);
}

export fn fizz_nif_mlirOperationStateAddResults(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]c.struct_MlirOperationState = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_c_struct_MlirOperationState, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirType = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirType, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationStateAddResultsWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirOperationStateGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirOperationStateGet(arg0, arg1);
}

export fn fizz_nif_mlirOperationStateGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirStringRef = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirStringRef, args[0]);
  var arg1: c.struct_MlirLocation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirLocation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperationState, @sizeOf(c.struct_MlirOperationState));

  const RType = c.struct_MlirOperationState;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirOperationStateGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirModuleFromOperationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirModuleFromOperation(arg0);
}

export fn fizz_nif_mlirModuleFromOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirOperation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirOperation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirModule, @sizeOf(c.struct_MlirModule));

  const RType = c.struct_MlirModule;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirModuleFromOperationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirModuleGetOperationWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirModuleGetOperation(arg0);
}

export fn fizz_nif_mlirModuleGetOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirModule = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirModule, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirOperation, @sizeOf(c.struct_MlirOperation));

  const RType = c.struct_MlirOperation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirModuleGetOperationWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirModuleDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirModuleDestroy(arg0);
}

export fn fizz_nif_mlirModuleDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirModule = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirModule, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirModuleDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirModuleGetBodyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirModuleGetBody(arg0);
}

export fn fizz_nif_mlirModuleGetBody(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirModule = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirModule, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirBlock, @sizeOf(c.struct_MlirBlock));

  const RType = c.struct_MlirBlock;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirModuleGetBodyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirModuleGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirModuleGetContext(arg0);
}

export fn fizz_nif_mlirModuleGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirModule = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirModule, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirModuleGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirModuleCreateParseWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirModuleCreateParse(arg0, arg1);
}

export fn fizz_nif_mlirModuleCreateParse(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirModule, @sizeOf(c.struct_MlirModule));

  const RType = c.struct_MlirModule;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirModuleCreateParseWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirModuleCreateEmptyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirModuleCreateEmpty(arg0);
}

export fn fizz_nif_mlirModuleCreateEmpty(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirModule, @sizeOf(c.struct_MlirModule));

  const RType = c.struct_MlirModule;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirModuleCreateEmptyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirLocationPrintWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirLocationPrint(arg0, arg1, arg2);
}

export fn fizz_nif_mlirLocationPrint(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: ?fn(c.struct_MlirStringRef, ?*anyopaque) callconv(.C) void = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void, args[1]);
  var arg2: ?*anyopaque = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_void_ptr, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLocationPrintWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirLocationEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirLocationEqual(arg0, arg1);
}

export fn fizz_nif_mlirLocationEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: c.struct_MlirLocation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirLocation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLocationEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirLocationGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirLocationGetContext(arg0);
}

export fn fizz_nif_mlirLocationGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLocationGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirLocationUnknownGetWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirLocationUnknownGet(arg0);
}

export fn fizz_nif_mlirLocationUnknownGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLocation, @sizeOf(c.struct_MlirLocation));

  const RType = c.struct_MlirLocation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLocationUnknownGetWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirLocationNameGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype) void {
  ret.* = c.mlirLocationNameGet(arg0, arg1, arg2);
}

export fn fizz_nif_mlirLocationNameGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: c.struct_MlirLocation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_struct_MlirLocation, args[2]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLocation, @sizeOf(c.struct_MlirLocation));

  const RType = c.struct_MlirLocation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLocationNameGetWrapper(obj, arg0, arg1, arg2);
  return e.enif_make_resource(env, ptr);
}

fn mlirLocationFusedGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirLocationFusedGet(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirLocationFusedGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: isize = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_isize, args[1]);
  var arg2: [*c]const c.struct_MlirLocation = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type__c_ptr_const_c_struct_MlirLocation, args[2]);
  var arg3: c.struct_MlirAttribute = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_struct_MlirAttribute, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLocation, @sizeOf(c.struct_MlirLocation));

  const RType = c.struct_MlirLocation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLocationFusedGetWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirLocationCallSiteGetWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirLocationCallSiteGet(arg0, arg1);
}

export fn fizz_nif_mlirLocationCallSiteGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirLocation = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirLocation, args[0]);
  var arg1: c.struct_MlirLocation = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirLocation, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLocation, @sizeOf(c.struct_MlirLocation));

  const RType = c.struct_MlirLocation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLocationCallSiteGetWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirLocationFileLineColGetWrapper(ret: anytype, arg0: anytype, arg1: anytype, arg2: anytype, arg3: anytype) void {
  ret.* = c.mlirLocationFileLineColGet(arg0, arg1, arg2, arg3);
}

export fn fizz_nif_mlirLocationFileLineColGet(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);
  var arg2: c_uint = undefined; arg2 = beam.fetch_resource(arg2, env, resource_type_c_uint, args[2]);
  var arg3: c_uint = undefined; arg3 = beam.fetch_resource(arg3, env, resource_type_c_uint, args[3]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirLocation, @sizeOf(c.struct_MlirLocation));

  const RType = c.struct_MlirLocation;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirLocationFileLineColGetWrapper(obj, arg0, arg1, arg2, arg3);
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectRegistryDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDialectRegistryDestroy(arg0);
}

export fn fizz_nif_mlirDialectRegistryDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDialectRegistry = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDialectRegistry, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectRegistryDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectRegistryCreateWrapper(ret: anytype, ) void {
  ret.* = c.mlirDialectRegistryCreate();
}

export fn fizz_nif_mlirDialectRegistryCreate(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialectRegistry, @sizeOf(c.struct_MlirDialectRegistry));

  const RType = c.struct_MlirDialectRegistry;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectRegistryCreateWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectGetNamespaceWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDialectGetNamespace(arg0);
}

export fn fizz_nif_mlirDialectGetNamespace(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDialect = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDialect, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectGetNamespaceWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirDialectEqual(arg0, arg1);
}

export fn fizz_nif_mlirDialectEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDialect = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDialect, args[0]);
  var arg1: c.struct_MlirDialect = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirDialect, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirDialectGetContextWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirDialectGetContext(arg0);
}

export fn fizz_nif_mlirDialectGetContext(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirDialect = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirDialect, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirDialectGetContextWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextIsRegisteredOperationWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirContextIsRegisteredOperation(arg0, arg1);
}

export fn fizz_nif_mlirContextIsRegisteredOperation(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextIsRegisteredOperationWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextEnableMultithreadingWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirContextEnableMultithreading(arg0, arg1);
}

export fn fizz_nif_mlirContextEnableMultithreading(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: bool = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_bool, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextEnableMultithreadingWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextGetOrLoadDialectWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirContextGetOrLoadDialect(arg0, arg1);
}

export fn fizz_nif_mlirContextGetOrLoadDialect(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirDialect, @sizeOf(c.struct_MlirDialect));

  const RType = c.struct_MlirDialect;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextGetOrLoadDialectWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextGetNumLoadedDialectsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirContextGetNumLoadedDialects(arg0);
}

export fn fizz_nif_mlirContextGetNumLoadedDialects(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextGetNumLoadedDialectsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextAppendDialectRegistryWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirContextAppendDialectRegistry(arg0, arg1);
}

export fn fizz_nif_mlirContextAppendDialectRegistry(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirDialectRegistry = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirDialectRegistry, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextAppendDialectRegistryWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextGetNumRegisteredDialectsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirContextGetNumRegisteredDialects(arg0);
}

export fn fizz_nif_mlirContextGetNumRegisteredDialects(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_isize, @sizeOf(isize));

  const RType = isize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextGetNumRegisteredDialectsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextGetAllowUnregisteredDialectsWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirContextGetAllowUnregisteredDialects(arg0);
}

export fn fizz_nif_mlirContextGetAllowUnregisteredDialects(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextGetAllowUnregisteredDialectsWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextSetAllowUnregisteredDialectsWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirContextSetAllowUnregisteredDialects(arg0, arg1);
}

export fn fizz_nif_mlirContextSetAllowUnregisteredDialects(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: bool = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_bool, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextSetAllowUnregisteredDialectsWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirContextDestroy(arg0);
}

export fn fizz_nif_mlirContextDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirContextEqual(arg0, arg1);
}

export fn fizz_nif_mlirContextEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirContext = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirContext, args[0]);
  var arg1: c.struct_MlirContext = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirContext, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirContextCreateWrapper(ret: anytype, ) void {
  ret.* = c.mlirContextCreate();
}

export fn fizz_nif_mlirContextCreate(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirContext, @sizeOf(c.struct_MlirContext));

  const RType = c.struct_MlirContext;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirContextCreateWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIDAllocatorAllocateTypeIDWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIDAllocatorAllocateTypeID(arg0);
}

export fn fizz_nif_mlirTypeIDAllocatorAllocateTypeID(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirTypeIDAllocator = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirTypeIDAllocator, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirTypeID, @sizeOf(c.struct_MlirTypeID));

  const RType = c.struct_MlirTypeID;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIDAllocatorAllocateTypeIDWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIDAllocatorDestroyWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIDAllocatorDestroy(arg0);
}

export fn fizz_nif_mlirTypeIDAllocatorDestroy(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirTypeIDAllocator = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirTypeIDAllocator, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_void, @sizeOf(void));

  const RType = void;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIDAllocatorDestroyWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIDAllocatorCreateWrapper(ret: anytype, ) void {
  ret.* = c.mlirTypeIDAllocatorCreate();
}

export fn fizz_nif_mlirTypeIDAllocatorCreate(env: beam.env, _: c_int, _: [*c] const beam.term) beam.term {
  
  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirTypeIDAllocator, @sizeOf(c.struct_MlirTypeIDAllocator));

  const RType = c.struct_MlirTypeIDAllocator;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIDAllocatorCreateWrapper(obj, );
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIDHashValueWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIDHashValue(arg0);
}

export fn fizz_nif_mlirTypeIDHashValue(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirTypeID = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirTypeID, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_usize, @sizeOf(usize));

  const RType = usize;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIDHashValueWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIDEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirTypeIDEqual(arg0, arg1);
}

export fn fizz_nif_mlirTypeIDEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirTypeID = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirTypeID, args[0]);
  var arg1: c.struct_MlirTypeID = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirTypeID, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIDEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirTypeIDCreateWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirTypeIDCreate(arg0);
}

export fn fizz_nif_mlirTypeIDCreate(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: ?*const anyopaque = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_const_void_ptr, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirTypeID, @sizeOf(c.struct_MlirTypeID));

  const RType = c.struct_MlirTypeID;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirTypeIDCreateWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}

fn mlirStringRefEqualWrapper(ret: anytype, arg0: anytype, arg1: anytype) void {
  ret.* = c.mlirStringRefEqual(arg0, arg1);
}

export fn fizz_nif_mlirStringRefEqual(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: c.struct_MlirStringRef = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type_c_struct_MlirStringRef, args[0]);
  var arg1: c.struct_MlirStringRef = undefined; arg1 = beam.fetch_resource(arg1, env, resource_type_c_struct_MlirStringRef, args[1]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_bool, @sizeOf(bool));

  const RType = bool;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirStringRefEqualWrapper(obj, arg0, arg1);
  return e.enif_make_resource(env, ptr);
}

fn mlirStringRefCreateFromCStringWrapper(ret: anytype, arg0: anytype) void {
  ret.* = c.mlirStringRefCreateFromCString(arg0);
}

export fn fizz_nif_mlirStringRefCreateFromCString(env: beam.env, _: c_int, args: [*c] const beam.term) beam.term {
  var arg0: [*c]const u8 = undefined; arg0 = beam.fetch_resource(arg0, env, resource_type__c_ptr_const_u8, args[0]);

  var ptr : ?*anyopaque = e.enif_alloc_resource(resource_type_c_struct_MlirStringRef, @sizeOf(c.struct_MlirStringRef));

  const RType = c.struct_MlirStringRef;
  var obj : *RType = undefined;

  if (ptr == null) {
    unreachable();
  } else {
    obj = @ptrCast(*RType, @alignCast(@alignOf(*RType), ptr));
  }
  mlirStringRefCreateFromCStringWrapper(obj, arg0);
  return e.enif_make_resource(env, ptr);
}


pub export fn __destroy__(_: beam.env, _: ?*anyopaque) void {
}
pub fn open_generated_resource_types(env: beam.env) void {
  resource_type_void_ptr = e.enif_open_resource_type(env, null, "resource_type_void_ptr", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_const_void_ptr = e.enif_open_resource_type(env, null, "resource_type_const_void_ptr", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__nullable_fn__nullable__pointer_anyopaque__callconv__C__void = e.enif_open_resource_type(env, null, "resource_type__nullable_fn__nullable__pointer_anyopaque__callconv__C__void", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__nullable_fn__nullable__pointer_anyopaque___isize___c_struct_MlirAffineMap__callconv__C__void = e.enif_open_resource_type(env, null, "resource_type__nullable_fn__nullable__pointer_anyopaque___isize___c_struct_MlirAffineMap__callconv__C__void", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__nullable_fn_c_struct_MlirDiagnostic____nullable__pointer_anyopaque__callconv__C__c_struct_MlirLogicalResult = e.enif_open_resource_type(env, null, "resource_type__nullable_fn_c_struct_MlirDiagnostic____nullable__pointer_anyopaque__callconv__C__c_struct_MlirLogicalResult", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__nullable_fn_c_struct_MlirOperation___bool____nullable__pointer_anyopaque__callconv__C__void = e.enif_open_resource_type(env, null, "resource_type__nullable_fn_c_struct_MlirOperation___bool____nullable__pointer_anyopaque__callconv__C__void", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void = e.enif_open_resource_type(env, null, "resource_type__nullable_fn_c_struct_MlirStringRef____nullable__pointer_anyopaque__callconv__C__void", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__nullable_fn_isize____c_ptr_c_struct_MlirType____nullable__pointer_anyopaque__callconv__C__void = e.enif_open_resource_type(env, null, "resource_type__nullable_fn_isize____c_ptr_c_struct_MlirType____nullable__pointer_anyopaque__callconv__C__void", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr__nullable__pointer_anyopaque = e.enif_open_resource_type(env, null, "resource_type__c_ptr__nullable__pointer_anyopaque", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_c_struct_MlirAffineExpr = e.enif_open_resource_type(env, null, "resource_type__c_ptr_c_struct_MlirAffineExpr", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_c_struct_MlirAffineMap = e.enif_open_resource_type(env, null, "resource_type__c_ptr_c_struct_MlirAffineMap", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_c_struct_MlirDialectHandle = e.enif_open_resource_type(env, null, "resource_type__c_ptr_c_struct_MlirDialectHandle", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_c_struct_MlirOperationState = e.enif_open_resource_type(env, null, "resource_type__c_ptr_c_struct_MlirOperationState", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_c_struct_MlirRegion = e.enif_open_resource_type(env, null, "resource_type__c_ptr_c_struct_MlirRegion", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_c_struct_MlirStringRef = e.enif_open_resource_type(env, null, "resource_type__c_ptr_c_struct_MlirStringRef", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_c_struct_MlirValue = e.enif_open_resource_type(env, null, "resource_type__c_ptr_c_struct_MlirValue", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_c_uint = e.enif_open_resource_type(env, null, "resource_type__c_ptr_c_uint", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_bool = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_bool", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirAffineExpr = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirAffineExpr", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirAttribute = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirAttribute", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirBlock = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirBlock", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirLocation = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirLocation", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirNamedAttribute = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirNamedAttribute", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirRegion = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirRegion", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirStringRef = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirStringRef", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirType = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirType", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_struct_MlirValue = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_struct_MlirValue", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_int = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_int", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_c_uint = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_c_uint", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_f32 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_f32", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_f64 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_f64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_i16 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_i16", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_i32 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_i32", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_i64 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_i64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_i8 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_i8", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_u16 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_u16", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_u32 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_u32", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_u64 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_u64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_const_u8 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_const_u8", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_f64 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_f64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_i64 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_i64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_isize = e.enif_open_resource_type(env, null, "resource_type__c_ptr_isize", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type__c_ptr_u64 = e.enif_open_resource_type(env, null, "resource_type__c_ptr_u64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_bool = e.enif_open_resource_type(env, null, "resource_type_bool", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirAffineExpr = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirAffineExpr", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirAffineMap = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirAffineMap", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirAttribute = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirAttribute", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirBlock = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirBlock", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirContext = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirContext", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirDiagnostic = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirDiagnostic", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirDialect = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirDialect", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirDialectHandle = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirDialectHandle", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirDialectRegistry = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirDialectRegistry", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirExecutionEngine = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirExecutionEngine", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirExternalPass = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirExternalPass", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirExternalPassCallbacks = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirExternalPassCallbacks", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirIdentifier = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirIdentifier", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirIntegerSet = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirIntegerSet", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirLocation = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirLocation", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirLogicalResult = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirLogicalResult", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirModule = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirModule", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirNamedAttribute = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirNamedAttribute", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirOpPassManager = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirOpPassManager", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirOpPrintingFlags = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirOpPrintingFlags", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirOperation = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirOperation", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirOperationState = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirOperationState", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirPass = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirPass", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirPassManager = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirPassManager", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirRegion = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirRegion", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirStringRef = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirStringRef", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirSymbolTable = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirSymbolTable", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirType = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirType", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirTypeID = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirTypeID", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirTypeIDAllocator = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirTypeIDAllocator", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_struct_MlirValue = e.enif_open_resource_type(env, null, "resource_type_c_struct_MlirValue", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_int = e.enif_open_resource_type(env, null, "resource_type_c_int", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_c_uint = e.enif_open_resource_type(env, null, "resource_type_c_uint", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_f32 = e.enif_open_resource_type(env, null, "resource_type_f32", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_f64 = e.enif_open_resource_type(env, null, "resource_type_f64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_i16 = e.enif_open_resource_type(env, null, "resource_type_i16", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_i32 = e.enif_open_resource_type(env, null, "resource_type_i32", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_i64 = e.enif_open_resource_type(env, null, "resource_type_i64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_i8 = e.enif_open_resource_type(env, null, "resource_type_i8", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_isize = e.enif_open_resource_type(env, null, "resource_type_isize", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_u16 = e.enif_open_resource_type(env, null, "resource_type_u16", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_u32 = e.enif_open_resource_type(env, null, "resource_type_u32", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_u64 = e.enif_open_resource_type(env, null, "resource_type_u64", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_u8 = e.enif_open_resource_type(env, null, "resource_type_u8", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_usize = e.enif_open_resource_type(env, null, "resource_type_usize", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);
  resource_type_void = e.enif_open_resource_type(env, null, "resource_type_void", __destroy__, e.ERL_NIF_RT_CREATE | e.ERL_NIF_RT_TAKEOVER, null);

}
pub export var generated_nifs = [_]e.ErlNifFunc{
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsViewOpGraph", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsViewOpGraph, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsViewOpGraph", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsViewOpGraph, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsTopologicalSort", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsTopologicalSort, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsTopologicalSort", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsTopologicalSort, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsSymbolPrivatize", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsSymbolPrivatize, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsSymbolPrivatize", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsSymbolPrivatize, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsSymbolDCE", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsSymbolDCE, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsSymbolDCE", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsSymbolDCE, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsStripDebugInfo", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsStripDebugInfo, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsStripDebugInfo", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsStripDebugInfo, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsSCCP", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsSCCP, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsSCCP", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsSCCP, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsPrintOpStats", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsPrintOpStats, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsPrintOpStats", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsPrintOpStats, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsLoopInvariantCodeMotion", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsLoopInvariantCodeMotion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsLoopInvariantCodeMotion", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsLoopInvariantCodeMotion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsLocationSnapshot", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsLocationSnapshot, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsLocationSnapshot", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsLocationSnapshot, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsInliner", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsInliner, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsInliner", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsInliner, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsControlFlowSink", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsControlFlowSink, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsControlFlowSink", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsControlFlowSink, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsCanonicalizer", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsCanonicalizer, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsCanonicalizer", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsCanonicalizer, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsCSE", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsCSE, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateTransformsCSE", .arity = 0, .fptr = fizz_nif_mlirCreateTransformsCSE, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterTransformsPasses", .arity = 0, .fptr = fizz_nif_mlirRegisterTransformsPasses, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirInferTypeOpInterfaceInferReturnTypes", .arity = 10, .fptr = fizz_nif_mlirInferTypeOpInterfaceInferReturnTypes, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirInferTypeOpInterfaceTypeID", .arity = 0, .fptr = fizz_nif_mlirInferTypeOpInterfaceTypeID, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationImplementsInterfaceStatic", .arity = 3, .fptr = fizz_nif_mlirOperationImplementsInterfaceStatic, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationImplementsInterface", .arity = 2, .fptr = fizz_nif_mlirOperationImplementsInterface, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetIsConstraintEq", .arity = 2, .fptr = fizz_nif_mlirIntegerSetIsConstraintEq, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGetConstraint", .arity = 2, .fptr = fizz_nif_mlirIntegerSetGetConstraint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGetNumInequalities", .arity = 1, .fptr = fizz_nif_mlirIntegerSetGetNumInequalities, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGetNumEqualities", .arity = 1, .fptr = fizz_nif_mlirIntegerSetGetNumEqualities, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGetNumConstraints", .arity = 1, .fptr = fizz_nif_mlirIntegerSetGetNumConstraints, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGetNumInputs", .arity = 1, .fptr = fizz_nif_mlirIntegerSetGetNumInputs, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGetNumSymbols", .arity = 1, .fptr = fizz_nif_mlirIntegerSetGetNumSymbols, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGetNumDims", .arity = 1, .fptr = fizz_nif_mlirIntegerSetGetNumDims, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetIsCanonicalEmpty", .arity = 1, .fptr = fizz_nif_mlirIntegerSetIsCanonicalEmpty, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetReplaceGet", .arity = 5, .fptr = fizz_nif_mlirIntegerSetReplaceGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGet", .arity = 6, .fptr = fizz_nif_mlirIntegerSetGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetEmptyGet", .arity = 3, .fptr = fizz_nif_mlirIntegerSetEmptyGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetDump", .arity = 1, .fptr = fizz_nif_mlirIntegerSetDump, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetPrint", .arity = 3, .fptr = fizz_nif_mlirIntegerSetPrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetEqual", .arity = 2, .fptr = fizz_nif_mlirIntegerSetEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerSetGetContext", .arity = 1, .fptr = fizz_nif_mlirIntegerSetGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirExecutionEngineDumpToObjectFile", .arity = 2, .fptr = fizz_nif_mlirExecutionEngineDumpToObjectFile, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirExecutionEngineRegisterSymbol", .arity = 3, .fptr = fizz_nif_mlirExecutionEngineRegisterSymbol, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirExecutionEngineLookup", .arity = 2, .fptr = fizz_nif_mlirExecutionEngineLookup, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirExecutionEngineLookupPacked", .arity = 2, .fptr = fizz_nif_mlirExecutionEngineLookupPacked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirExecutionEngineInvokePacked", .arity = 3, .fptr = fizz_nif_mlirExecutionEngineInvokePacked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirExecutionEngineDestroy", .arity = 1, .fptr = fizz_nif_mlirExecutionEngineDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirExecutionEngineCreate", .arity = 4, .fptr = fizz_nif_mlirExecutionEngineCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__tensor__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__tensor__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterSparseTensorSparsification", .arity = 0, .fptr = fizz_nif_mlirRegisterSparseTensorSparsification, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateSparseTensorSparsification", .arity = 0, .fptr = fizz_nif_mlirCreateSparseTensorSparsification, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterSparseTensorSparseTensorConversion", .arity = 0, .fptr = fizz_nif_mlirRegisterSparseTensorSparseTensorConversion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateSparseTensorSparseTensorConversion", .arity = 0, .fptr = fizz_nif_mlirCreateSparseTensorSparseTensorConversion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterSparseTensorPasses", .arity = 0, .fptr = fizz_nif_mlirRegisterSparseTensorPasses, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseTensorEncodingAttrGetIndexBitWidth", .arity = 1, .fptr = fizz_nif_mlirSparseTensorEncodingAttrGetIndexBitWidth, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseTensorEncodingAttrGetPointerBitWidth", .arity = 1, .fptr = fizz_nif_mlirSparseTensorEncodingAttrGetPointerBitWidth, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseTensorEncodingAttrGetDimOrdering", .arity = 1, .fptr = fizz_nif_mlirSparseTensorEncodingAttrGetDimOrdering, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseTensorEncodingAttrGetDimLevelType", .arity = 2, .fptr = fizz_nif_mlirSparseTensorEncodingAttrGetDimLevelType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseTensorEncodingGetNumDimLevelTypes", .arity = 1, .fptr = fizz_nif_mlirSparseTensorEncodingGetNumDimLevelTypes, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseTensorEncodingAttrGet", .arity = 6, .fptr = fizz_nif_mlirSparseTensorEncodingAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsASparseTensorEncodingAttr", .arity = 1, .fptr = fizz_nif_mlirAttributeIsASparseTensorEncodingAttr, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__sparse_tensor__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__sparse_tensor__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__shape__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__shape__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__scf__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__scf__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCalibratedQuantizedTypeGetMax", .arity = 1, .fptr = fizz_nif_mlirCalibratedQuantizedTypeGetMax, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCalibratedQuantizedTypeGetMin", .arity = 1, .fptr = fizz_nif_mlirCalibratedQuantizedTypeGetMin, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCalibratedQuantizedTypeGet", .arity = 3, .fptr = fizz_nif_mlirCalibratedQuantizedTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsACalibratedQuantizedType", .arity = 1, .fptr = fizz_nif_mlirTypeIsACalibratedQuantizedType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedPerAxisTypeIsFixedPoint", .arity = 1, .fptr = fizz_nif_mlirUniformQuantizedPerAxisTypeIsFixedPoint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedPerAxisTypeGetQuantizedDimension", .arity = 1, .fptr = fizz_nif_mlirUniformQuantizedPerAxisTypeGetQuantizedDimension, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedPerAxisTypeGetZeroPoint", .arity = 2, .fptr = fizz_nif_mlirUniformQuantizedPerAxisTypeGetZeroPoint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedPerAxisTypeGetScale", .arity = 2, .fptr = fizz_nif_mlirUniformQuantizedPerAxisTypeGetScale, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedPerAxisTypeGetNumDims", .arity = 1, .fptr = fizz_nif_mlirUniformQuantizedPerAxisTypeGetNumDims, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedPerAxisTypeGet", .arity = 9, .fptr = fizz_nif_mlirUniformQuantizedPerAxisTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAUniformQuantizedPerAxisType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAUniformQuantizedPerAxisType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedTypeIsFixedPoint", .arity = 1, .fptr = fizz_nif_mlirUniformQuantizedTypeIsFixedPoint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedTypeGetZeroPoint", .arity = 1, .fptr = fizz_nif_mlirUniformQuantizedTypeGetZeroPoint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedTypeGetScale", .arity = 1, .fptr = fizz_nif_mlirUniformQuantizedTypeGetScale, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUniformQuantizedTypeGet", .arity = 7, .fptr = fizz_nif_mlirUniformQuantizedTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAUniformQuantizedType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAUniformQuantizedType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAnyQuantizedTypeGet", .arity = 5, .fptr = fizz_nif_mlirAnyQuantizedTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAAnyQuantizedType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAAnyQuantizedType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeCastExpressedToStorageType", .arity = 2, .fptr = fizz_nif_mlirQuantizedTypeCastExpressedToStorageType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeCastToExpressedType", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeCastToExpressedType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeCastFromExpressedType", .arity = 2, .fptr = fizz_nif_mlirQuantizedTypeCastFromExpressedType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeCastToStorageType", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeCastToStorageType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeCastFromStorageType", .arity = 2, .fptr = fizz_nif_mlirQuantizedTypeCastFromStorageType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetQuantizedElementType", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeGetQuantizedElementType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeIsCompatibleExpressedType", .arity = 2, .fptr = fizz_nif_mlirQuantizedTypeIsCompatibleExpressedType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetStorageTypeIntegralWidth", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeGetStorageTypeIntegralWidth, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetStorageTypeMax", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeGetStorageTypeMax, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetStorageTypeMin", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeGetStorageTypeMin, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetStorageType", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeGetStorageType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeIsSigned", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeIsSigned, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetFlags", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeGetFlags, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetExpressedType", .arity = 1, .fptr = fizz_nif_mlirQuantizedTypeGetExpressedType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetDefaultMaximumForInteger", .arity = 2, .fptr = fizz_nif_mlirQuantizedTypeGetDefaultMaximumForInteger, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetDefaultMinimumForInteger", .arity = 2, .fptr = fizz_nif_mlirQuantizedTypeGetDefaultMinimumForInteger, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirQuantizedTypeGetSignedFlag", .arity = 0, .fptr = fizz_nif_mlirQuantizedTypeGetSignedFlag, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAQuantizedType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAQuantizedType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__quant__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__quant__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPDLValueTypeGet", .arity = 1, .fptr = fizz_nif_mlirPDLValueTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAPDLValueType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAPDLValueType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPDLTypeTypeGet", .arity = 1, .fptr = fizz_nif_mlirPDLTypeTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAPDLTypeType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAPDLTypeType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPDLRangeTypeGetElementType", .arity = 1, .fptr = fizz_nif_mlirPDLRangeTypeGetElementType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPDLRangeTypeGet", .arity = 1, .fptr = fizz_nif_mlirPDLRangeTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAPDLRangeType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAPDLRangeType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPDLOperationTypeGet", .arity = 1, .fptr = fizz_nif_mlirPDLOperationTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAPDLOperationType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAPDLOperationType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPDLAttributeTypeGet", .arity = 1, .fptr = fizz_nif_mlirPDLAttributeTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAPDLAttributeType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAPDLAttributeType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAPDLType", .arity = 1, .fptr = fizz_nif_mlirTypeIsAPDLType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__pdl__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__pdl__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgTiling", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgTiling, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgTiling", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgTiling, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyVectorizePass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyVectorizePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyVectorizePass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyVectorizePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyTilePass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyTilePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyTilePass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyTilePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyTileAndFusePass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyTileAndFusePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyTileAndFusePass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyTileAndFusePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyRemoveMarkersPass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyRemoveMarkersPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyRemoveMarkersPass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyRemoveMarkersPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyPromotePass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyPromotePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyPromotePass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyPromotePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyPeelPass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyPeelPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyPeelPass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyPeelPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyPadPass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyPadPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyPadPass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyPadPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyLowerVectorsPass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyLowerVectorsPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyLowerVectorsPass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyLowerVectorsPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyInterchangePass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyInterchangePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyInterchangePass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyInterchangePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyGeneralizePass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyGeneralizePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyGeneralizePass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyGeneralizePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyEnablePass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyEnablePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyEnablePass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyEnablePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgStrategyDecomposePass", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgStrategyDecomposePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgStrategyDecomposePass", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgStrategyDecomposePass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgPromotion", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgPromotion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgPromotion", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgPromotion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgNamedOpConversion", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgNamedOpConversion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgNamedOpConversion", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgNamedOpConversion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgLowerToParallelLoops", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgLowerToParallelLoops, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgLowerToParallelLoops", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgLowerToParallelLoops, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgLowerToLoops", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgLowerToLoops, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgLowerToLoops", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgLowerToLoops, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgLowerToAffineLoops", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgLowerToAffineLoops, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgLowerToAffineLoops", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgLowerToAffineLoops, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgInlineScalarOperands", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgInlineScalarOperands, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgInlineScalarOperands", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgInlineScalarOperands, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgInitTensorToAllocTensor", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgInitTensorToAllocTensor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgInitTensorToAllocTensor", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgInitTensorToAllocTensor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgGeneralization", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgGeneralization, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgGeneralization", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgGeneralization, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgFoldUnitExtentDims", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgFoldUnitExtentDims, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgFoldUnitExtentDims", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgFoldUnitExtentDims, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgElementwiseOpFusion", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgElementwiseOpFusion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgElementwiseOpFusion", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgElementwiseOpFusion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgDetensorize", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgDetensorize, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgDetensorize", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgDetensorize, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgLinalgBufferize", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgLinalgBufferize, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgLinalgBufferize", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgLinalgBufferize, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgConvertElementwiseToLinalg", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgConvertElementwiseToLinalg, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateLinalgConvertElementwiseToLinalg", .arity = 0, .fptr = fizz_nif_mlirCreateLinalgConvertElementwiseToLinalg, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterLinalgPasses", .arity = 0, .fptr = fizz_nif_mlirRegisterLinalgPasses, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__linalg__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__linalg__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLinalgFillBuiltinNamedOpRegion", .arity = 1, .fptr = fizz_nif_mlirLinalgFillBuiltinNamedOpRegion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLLVMStructTypeLiteralGet", .arity = 4, .fptr = fizz_nif_mlirLLVMStructTypeLiteralGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLLVMFunctionTypeGet", .arity = 4, .fptr = fizz_nif_mlirLLVMFunctionTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLLVMArrayTypeGet", .arity = 2, .fptr = fizz_nif_mlirLLVMArrayTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLLVMVoidTypeGet", .arity = 1, .fptr = fizz_nif_mlirLLVMVoidTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLLVMPointerTypeGet", .arity = 2, .fptr = fizz_nif_mlirLLVMPointerTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__llvm__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__llvm__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterGPUGpuMapParallelLoopsPass", .arity = 0, .fptr = fizz_nif_mlirRegisterGPUGpuMapParallelLoopsPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateGPUGpuMapParallelLoopsPass", .arity = 0, .fptr = fizz_nif_mlirCreateGPUGpuMapParallelLoopsPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterGPUGpuLaunchSinkIndexComputations", .arity = 0, .fptr = fizz_nif_mlirRegisterGPUGpuLaunchSinkIndexComputations, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateGPUGpuLaunchSinkIndexComputations", .arity = 0, .fptr = fizz_nif_mlirCreateGPUGpuLaunchSinkIndexComputations, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterGPUGpuKernelOutlining", .arity = 0, .fptr = fizz_nif_mlirRegisterGPUGpuKernelOutlining, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateGPUGpuKernelOutlining", .arity = 0, .fptr = fizz_nif_mlirCreateGPUGpuKernelOutlining, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterGPUGpuAsyncRegionPass", .arity = 0, .fptr = fizz_nif_mlirRegisterGPUGpuAsyncRegionPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateGPUGpuAsyncRegionPass", .arity = 0, .fptr = fizz_nif_mlirCreateGPUGpuAsyncRegionPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterGPUPasses", .arity = 0, .fptr = fizz_nif_mlirRegisterGPUPasses, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__gpu__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__gpu__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__func__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__func__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__elixir__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__elixir__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__cf__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__cf__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAsyncAsyncToAsyncRuntime", .arity = 0, .fptr = fizz_nif_mlirRegisterAsyncAsyncToAsyncRuntime, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateAsyncAsyncToAsyncRuntime", .arity = 0, .fptr = fizz_nif_mlirCreateAsyncAsyncToAsyncRuntime, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAsyncAsyncRuntimeRefCountingOpt", .arity = 0, .fptr = fizz_nif_mlirRegisterAsyncAsyncRuntimeRefCountingOpt, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateAsyncAsyncRuntimeRefCountingOpt", .arity = 0, .fptr = fizz_nif_mlirCreateAsyncAsyncRuntimeRefCountingOpt, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAsyncAsyncRuntimeRefCounting", .arity = 0, .fptr = fizz_nif_mlirRegisterAsyncAsyncRuntimeRefCounting, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateAsyncAsyncRuntimeRefCounting", .arity = 0, .fptr = fizz_nif_mlirCreateAsyncAsyncRuntimeRefCounting, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting", .arity = 0, .fptr = fizz_nif_mlirRegisterAsyncAsyncRuntimePolicyBasedRefCounting, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting", .arity = 0, .fptr = fizz_nif_mlirCreateAsyncAsyncRuntimePolicyBasedRefCounting, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAsyncAsyncParallelFor", .arity = 0, .fptr = fizz_nif_mlirRegisterAsyncAsyncParallelFor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateAsyncAsyncParallelFor", .arity = 0, .fptr = fizz_nif_mlirCreateAsyncAsyncParallelFor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAsyncPasses", .arity = 0, .fptr = fizz_nif_mlirRegisterAsyncPasses, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirGetDialectHandle__async__", .arity = 0, .fptr = fizz_nif_mlirGetDialectHandle__async__, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirEmitError", .arity = 2, .fptr = fizz_nif_mlirEmitError, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextDetachDiagnosticHandler", .arity = 2, .fptr = fizz_nif_mlirContextDetachDiagnosticHandler, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextAttachDiagnosticHandler", .arity = 4, .fptr = fizz_nif_mlirContextAttachDiagnosticHandler, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDiagnosticGetNote", .arity = 2, .fptr = fizz_nif_mlirDiagnosticGetNote, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDiagnosticGetNumNotes", .arity = 1, .fptr = fizz_nif_mlirDiagnosticGetNumNotes, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDiagnosticGetSeverity", .arity = 1, .fptr = fizz_nif_mlirDiagnosticGetSeverity, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDiagnosticGetLocation", .arity = 1, .fptr = fizz_nif_mlirDiagnosticGetLocation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDiagnosticPrint", .arity = 3, .fptr = fizz_nif_mlirDiagnosticPrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIsGlobalDebugEnabled", .arity = 0, .fptr = fizz_nif_mlirIsGlobalDebugEnabled, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirEnableGlobalDebug", .arity = 1, .fptr = fizz_nif_mlirEnableGlobalDebug, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionTosaToTensor", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionTosaToTensor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionTosaToTensor", .arity = 0, .fptr = fizz_nif_mlirCreateConversionTosaToTensor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionTosaToSCF", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionTosaToSCF, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionTosaToSCF", .arity = 0, .fptr = fizz_nif_mlirCreateConversionTosaToSCF, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionTosaToLinalgNamed", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionTosaToLinalgNamed, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionTosaToLinalgNamed", .arity = 0, .fptr = fizz_nif_mlirCreateConversionTosaToLinalgNamed, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionTosaToLinalg", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionTosaToLinalg, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionTosaToLinalg", .arity = 0, .fptr = fizz_nif_mlirCreateConversionTosaToLinalg, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionTosaToArith", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionTosaToArith, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionTosaToArith", .arity = 0, .fptr = fizz_nif_mlirCreateConversionTosaToArith, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionSCFToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionSCFToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionSCFToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionSCFToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionSCFToControlFlow", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionSCFToControlFlow, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionSCFToControlFlow", .arity = 0, .fptr = fizz_nif_mlirCreateConversionSCFToControlFlow, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionReconcileUnrealizedCasts", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionReconcileUnrealizedCasts, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionReconcileUnrealizedCasts", .arity = 0, .fptr = fizz_nif_mlirCreateConversionReconcileUnrealizedCasts, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionLowerHostCodeToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionLowerHostCodeToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionLowerHostCodeToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionLowerHostCodeToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionGpuToLLVMConversionPass", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionGpuToLLVMConversionPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionGpuToLLVMConversionPass", .arity = 0, .fptr = fizz_nif_mlirCreateConversionGpuToLLVMConversionPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCalls", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertVulkanLaunchFuncToVulkanCalls, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertVectorToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertVectorToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertVectorToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertVectorToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertVectorToSCF", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertVectorToSCF, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertVectorToSCF", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertVectorToSCF, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertVectorToROCDL", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertVectorToROCDL, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertVectorToROCDL", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertVectorToROCDL, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertVectorToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertVectorToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertVectorToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertVectorToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertVectorToGPU", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertVectorToGPU, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertVectorToGPU", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertVectorToGPU, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertTensorToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertTensorToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertTensorToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertTensorToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertTensorToLinalg", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertTensorToLinalg, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertTensorToLinalg", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertTensorToLinalg, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertShapeToStandard", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertShapeToStandard, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertShapeToStandard", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertShapeToStandard, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertShapeConstraints", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertShapeConstraints, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertShapeConstraints", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertShapeConstraints, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertSPIRVToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertSPIRVToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertSPIRVToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertSPIRVToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertSCFToOpenMP", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertSCFToOpenMP, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertSCFToOpenMP", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertSCFToOpenMP, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertParallelLoopToGpu", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertParallelLoopToGpu, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertParallelLoopToGpu", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertParallelLoopToGpu, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertPDLToPDLInterp", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertPDLToPDLInterp, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertPDLToPDLInterp", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertPDLToPDLInterp, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertOpenMPToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertOpenMPToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertOpenMPToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertOpenMPToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertOpenACCToSCF", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertOpenACCToSCF, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertOpenACCToSCF", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertOpenACCToSCF, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertOpenACCToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertOpenACCToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertOpenACCToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertOpenACCToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertNVGPUToNVVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertNVGPUToNVVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertNVGPUToNVVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertNVGPUToNVVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertMemRefToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertMemRefToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertMemRefToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertMemRefToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertMemRefToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertMemRefToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertMemRefToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertMemRefToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertMathToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertMathToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertMathToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertMathToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertMathToLibm", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertMathToLibm, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertMathToLibm", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertMathToLibm, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertMathToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertMathToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertMathToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertMathToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertLinalgToStandard", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertLinalgToStandard, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertLinalgToStandard", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertLinalgToStandard, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertLinalgToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertLinalgToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertLinalgToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertLinalgToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertLinalgToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertLinalgToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertLinalgToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertLinalgToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertGpuOpsToROCDLOps", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertGpuOpsToROCDLOps, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertGpuOpsToROCDLOps", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertGpuOpsToROCDLOps, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertGpuOpsToNVVMOps", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertGpuOpsToNVVMOps, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertGpuOpsToNVVMOps", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertGpuOpsToNVVMOps, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertGpuLaunchFuncToVulkanLaunchFunc, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertGPUToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertGPUToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertGPUToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertGPUToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertFuncToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertFuncToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertFuncToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertFuncToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertFuncToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertFuncToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertFuncToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertFuncToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertControlFlowToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertControlFlowToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertControlFlowToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertControlFlowToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertControlFlowToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertControlFlowToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertControlFlowToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertControlFlowToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertComplexToStandard", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertComplexToStandard, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertComplexToStandard", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertComplexToStandard, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertComplexToLibm", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertComplexToLibm, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertComplexToLibm", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertComplexToLibm, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertComplexToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertComplexToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertComplexToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertComplexToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertBufferizationToMemRef", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertBufferizationToMemRef, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertBufferizationToMemRef", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertBufferizationToMemRef, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertAsyncToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertAsyncToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertAsyncToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertAsyncToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertArmNeon2dToIntr", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertArmNeon2dToIntr, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertArmNeon2dToIntr", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertArmNeon2dToIntr, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertArithmeticToSPIRV", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertArithmeticToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertArithmeticToSPIRV", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertArithmeticToSPIRV, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertArithmeticToLLVM", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertArithmeticToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertArithmeticToLLVM", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertArithmeticToLLVM, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertAffineToStandard", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertAffineToStandard, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertAffineToStandard", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertAffineToStandard, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertAffineForToGPU", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertAffineForToGPU, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertAffineForToGPU", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertAffineForToGPU, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionConvertAMDGPUToROCDL", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionConvertAMDGPUToROCDL, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateConversionConvertAMDGPUToROCDL", .arity = 0, .fptr = fizz_nif_mlirCreateConversionConvertAMDGPUToROCDL, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterConversionPasses", .arity = 0, .fptr = fizz_nif_mlirRegisterConversionPasses, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpaqueTypeGetData", .arity = 1, .fptr = fizz_nif_mlirOpaqueTypeGetData, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpaqueTypeGetDialectNamespace", .arity = 1, .fptr = fizz_nif_mlirOpaqueTypeGetDialectNamespace, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpaqueTypeGet", .arity = 3, .fptr = fizz_nif_mlirOpaqueTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAOpaque", .arity = 1, .fptr = fizz_nif_mlirTypeIsAOpaque, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFunctionTypeGetResult", .arity = 2, .fptr = fizz_nif_mlirFunctionTypeGetResult, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFunctionTypeGetInput", .arity = 2, .fptr = fizz_nif_mlirFunctionTypeGetInput, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFunctionTypeGetNumResults", .arity = 1, .fptr = fizz_nif_mlirFunctionTypeGetNumResults, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFunctionTypeGetNumInputs", .arity = 1, .fptr = fizz_nif_mlirFunctionTypeGetNumInputs, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFunctionTypeGet", .arity = 5, .fptr = fizz_nif_mlirFunctionTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAFunction", .arity = 1, .fptr = fizz_nif_mlirTypeIsAFunction, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTupleTypeGetType", .arity = 2, .fptr = fizz_nif_mlirTupleTypeGetType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTupleTypeGetNumTypes", .arity = 1, .fptr = fizz_nif_mlirTupleTypeGetNumTypes, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTupleTypeGet", .arity = 3, .fptr = fizz_nif_mlirTupleTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsATuple", .arity = 1, .fptr = fizz_nif_mlirTypeIsATuple, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUnrankedMemrefGetMemorySpace", .arity = 1, .fptr = fizz_nif_mlirUnrankedMemrefGetMemorySpace, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirMemRefTypeGetMemorySpace", .arity = 1, .fptr = fizz_nif_mlirMemRefTypeGetMemorySpace, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirMemRefTypeGetAffineMap", .arity = 1, .fptr = fizz_nif_mlirMemRefTypeGetAffineMap, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirMemRefTypeGetLayout", .arity = 1, .fptr = fizz_nif_mlirMemRefTypeGetLayout, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUnrankedMemRefTypeGetChecked", .arity = 3, .fptr = fizz_nif_mlirUnrankedMemRefTypeGetChecked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUnrankedMemRefTypeGet", .arity = 2, .fptr = fizz_nif_mlirUnrankedMemRefTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirMemRefTypeContiguousGetChecked", .arity = 5, .fptr = fizz_nif_mlirMemRefTypeContiguousGetChecked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirMemRefTypeContiguousGet", .arity = 4, .fptr = fizz_nif_mlirMemRefTypeContiguousGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirMemRefTypeGetChecked", .arity = 6, .fptr = fizz_nif_mlirMemRefTypeGetChecked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirMemRefTypeGet", .arity = 5, .fptr = fizz_nif_mlirMemRefTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAUnrankedMemRef", .arity = 1, .fptr = fizz_nif_mlirTypeIsAUnrankedMemRef, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAMemRef", .arity = 1, .fptr = fizz_nif_mlirTypeIsAMemRef, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUnrankedTensorTypeGetChecked", .arity = 2, .fptr = fizz_nif_mlirUnrankedTensorTypeGetChecked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUnrankedTensorTypeGet", .arity = 1, .fptr = fizz_nif_mlirUnrankedTensorTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRankedTensorTypeGetEncoding", .arity = 1, .fptr = fizz_nif_mlirRankedTensorTypeGetEncoding, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRankedTensorTypeGetChecked", .arity = 5, .fptr = fizz_nif_mlirRankedTensorTypeGetChecked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRankedTensorTypeGet", .arity = 4, .fptr = fizz_nif_mlirRankedTensorTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAUnrankedTensor", .arity = 1, .fptr = fizz_nif_mlirTypeIsAUnrankedTensor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsARankedTensor", .arity = 1, .fptr = fizz_nif_mlirTypeIsARankedTensor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsATensor", .arity = 1, .fptr = fizz_nif_mlirTypeIsATensor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirVectorTypeGetChecked", .arity = 4, .fptr = fizz_nif_mlirVectorTypeGetChecked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirVectorTypeGet", .arity = 3, .fptr = fizz_nif_mlirVectorTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAVector", .arity = 1, .fptr = fizz_nif_mlirTypeIsAVector, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirShapedTypeIsDynamicStrideOrOffset", .arity = 1, .fptr = fizz_nif_mlirShapedTypeIsDynamicStrideOrOffset, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirShapedTypeIsDynamicSize", .arity = 1, .fptr = fizz_nif_mlirShapedTypeIsDynamicSize, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirShapedTypeGetDimSize", .arity = 2, .fptr = fizz_nif_mlirShapedTypeGetDimSize, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirShapedTypeIsDynamicDim", .arity = 2, .fptr = fizz_nif_mlirShapedTypeIsDynamicDim, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirShapedTypeHasStaticShape", .arity = 1, .fptr = fizz_nif_mlirShapedTypeHasStaticShape, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirShapedTypeGetRank", .arity = 1, .fptr = fizz_nif_mlirShapedTypeGetRank, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirShapedTypeHasRank", .arity = 1, .fptr = fizz_nif_mlirShapedTypeHasRank, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirShapedTypeGetElementType", .arity = 1, .fptr = fizz_nif_mlirShapedTypeGetElementType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAShaped", .arity = 1, .fptr = fizz_nif_mlirTypeIsAShaped, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirComplexTypeGetElementType", .arity = 1, .fptr = fizz_nif_mlirComplexTypeGetElementType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirComplexTypeGet", .arity = 1, .fptr = fizz_nif_mlirComplexTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAComplex", .arity = 1, .fptr = fizz_nif_mlirTypeIsAComplex, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirNoneTypeGet", .arity = 1, .fptr = fizz_nif_mlirNoneTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsANone", .arity = 1, .fptr = fizz_nif_mlirTypeIsANone, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirF64TypeGet", .arity = 1, .fptr = fizz_nif_mlirF64TypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAF64", .arity = 1, .fptr = fizz_nif_mlirTypeIsAF64, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirF32TypeGet", .arity = 1, .fptr = fizz_nif_mlirF32TypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAF32", .arity = 1, .fptr = fizz_nif_mlirTypeIsAF32, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirF16TypeGet", .arity = 1, .fptr = fizz_nif_mlirF16TypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAF16", .arity = 1, .fptr = fizz_nif_mlirTypeIsAF16, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBF16TypeGet", .arity = 1, .fptr = fizz_nif_mlirBF16TypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsABF16", .arity = 1, .fptr = fizz_nif_mlirTypeIsABF16, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIndexTypeGet", .arity = 1, .fptr = fizz_nif_mlirIndexTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAIndex", .arity = 1, .fptr = fizz_nif_mlirTypeIsAIndex, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerTypeIsUnsigned", .arity = 1, .fptr = fizz_nif_mlirIntegerTypeIsUnsigned, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerTypeIsSigned", .arity = 1, .fptr = fizz_nif_mlirIntegerTypeIsSigned, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerTypeIsSignless", .arity = 1, .fptr = fizz_nif_mlirIntegerTypeIsSignless, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerTypeGetWidth", .arity = 1, .fptr = fizz_nif_mlirIntegerTypeGetWidth, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerTypeUnsignedGet", .arity = 2, .fptr = fizz_nif_mlirIntegerTypeUnsignedGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerTypeSignedGet", .arity = 2, .fptr = fizz_nif_mlirIntegerTypeSignedGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerTypeGet", .arity = 2, .fptr = fizz_nif_mlirIntegerTypeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIsAInteger", .arity = 1, .fptr = fizz_nif_mlirTypeIsAInteger, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseElementsAttrGetValues", .arity = 1, .fptr = fizz_nif_mlirSparseElementsAttrGetValues, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseElementsAttrGetIndices", .arity = 1, .fptr = fizz_nif_mlirSparseElementsAttrGetIndices, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSparseElementsAttribute", .arity = 3, .fptr = fizz_nif_mlirSparseElementsAttribute, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsASparseElements", .arity = 1, .fptr = fizz_nif_mlirAttributeIsASparseElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAOpaqueElements", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAOpaqueElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetRawData", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetRawData, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetStringValue", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetStringValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetDoubleValue", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetDoubleValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetFloatValue", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetFloatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetUInt64Value", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetUInt64Value, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetInt64Value", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetInt64Value, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetUInt32Value", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetUInt32Value, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetInt32Value", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetInt32Value, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetUInt16Value", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetUInt16Value, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetInt16Value", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetInt16Value, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetUInt8Value", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetUInt8Value, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetInt8Value", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetInt8Value, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetBoolValue", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrGetBoolValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetStringSplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetStringSplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetDoubleSplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetDoubleSplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetFloatSplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetFloatSplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetUInt64SplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetUInt64SplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetInt64SplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetInt64SplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetUInt32SplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetUInt32SplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetInt32SplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetInt32SplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetUInt8SplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetUInt8SplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetInt8SplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetInt8SplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetBoolSplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetBoolSplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGetSplatValue", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrGetSplatValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrIsSplat", .arity = 1, .fptr = fizz_nif_mlirDenseElementsAttrIsSplat, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrReshapeGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrReshapeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrStringGet", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrStringGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrBFloat16Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrBFloat16Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrDoubleGet", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrDoubleGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrFloatGet", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrFloatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrInt64Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrInt64Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrUInt64Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrUInt64Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrInt32Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrInt32Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrUInt32Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrUInt32Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrInt16Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrInt16Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrUInt16Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrUInt16Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrInt8Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrInt8Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrUInt8Get", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrUInt8Get, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrBoolGet", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrBoolGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrDoubleSplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrDoubleSplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrFloatSplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrFloatSplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrInt64SplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrInt64SplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrUInt64SplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrUInt64SplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrInt32SplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrInt32SplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrUInt32SplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrUInt32SplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrInt8SplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrInt8SplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrUInt8SplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrUInt8SplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrBoolSplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrBoolSplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrSplatGet", .arity = 2, .fptr = fizz_nif_mlirDenseElementsAttrSplatGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrRawBufferGet", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrRawBufferGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDenseElementsAttrGet", .arity = 3, .fptr = fizz_nif_mlirDenseElementsAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsADenseFPElements", .arity = 1, .fptr = fizz_nif_mlirAttributeIsADenseFPElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsADenseIntElements", .arity = 1, .fptr = fizz_nif_mlirAttributeIsADenseIntElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsADenseElements", .arity = 1, .fptr = fizz_nif_mlirAttributeIsADenseElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirElementsAttrGetNumElements", .arity = 1, .fptr = fizz_nif_mlirElementsAttrGetNumElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirElementsAttrIsValidIndex", .arity = 3, .fptr = fizz_nif_mlirElementsAttrIsValidIndex, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirElementsAttrGetValue", .arity = 3, .fptr = fizz_nif_mlirElementsAttrGetValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAElements", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirUnitAttrGet", .arity = 1, .fptr = fizz_nif_mlirUnitAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAUnit", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAUnit, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeAttrGetValue", .arity = 1, .fptr = fizz_nif_mlirTypeAttrGetValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeAttrGet", .arity = 1, .fptr = fizz_nif_mlirTypeAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAType", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFlatSymbolRefAttrGetValue", .arity = 1, .fptr = fizz_nif_mlirFlatSymbolRefAttrGetValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFlatSymbolRefAttrGet", .arity = 2, .fptr = fizz_nif_mlirFlatSymbolRefAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAFlatSymbolRef", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAFlatSymbolRef, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolRefAttrGetNestedReference", .arity = 2, .fptr = fizz_nif_mlirSymbolRefAttrGetNestedReference, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolRefAttrGetNumNestedReferences", .arity = 1, .fptr = fizz_nif_mlirSymbolRefAttrGetNumNestedReferences, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolRefAttrGetLeafReference", .arity = 1, .fptr = fizz_nif_mlirSymbolRefAttrGetLeafReference, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolRefAttrGetRootReference", .arity = 1, .fptr = fizz_nif_mlirSymbolRefAttrGetRootReference, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolRefAttrGet", .arity = 4, .fptr = fizz_nif_mlirSymbolRefAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsASymbolRef", .arity = 1, .fptr = fizz_nif_mlirAttributeIsASymbolRef, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirStringAttrGetValue", .arity = 1, .fptr = fizz_nif_mlirStringAttrGetValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirStringAttrTypedGet", .arity = 2, .fptr = fizz_nif_mlirStringAttrTypedGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirStringAttrGet", .arity = 2, .fptr = fizz_nif_mlirStringAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAString", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAString, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpaqueAttrGetData", .arity = 1, .fptr = fizz_nif_mlirOpaqueAttrGetData, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpaqueAttrGetDialectNamespace", .arity = 1, .fptr = fizz_nif_mlirOpaqueAttrGetDialectNamespace, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpaqueAttrGet", .arity = 5, .fptr = fizz_nif_mlirOpaqueAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAOpaque", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAOpaque, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAIntegerSet", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAIntegerSet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBoolAttrGetValue", .arity = 1, .fptr = fizz_nif_mlirBoolAttrGetValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBoolAttrGet", .arity = 2, .fptr = fizz_nif_mlirBoolAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsABool", .arity = 1, .fptr = fizz_nif_mlirAttributeIsABool, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerAttrGetValueUInt", .arity = 1, .fptr = fizz_nif_mlirIntegerAttrGetValueUInt, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerAttrGetValueSInt", .arity = 1, .fptr = fizz_nif_mlirIntegerAttrGetValueSInt, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerAttrGetValueInt", .arity = 1, .fptr = fizz_nif_mlirIntegerAttrGetValueInt, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIntegerAttrGet", .arity = 2, .fptr = fizz_nif_mlirIntegerAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAInteger", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAInteger, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFloatAttrGetValueDouble", .arity = 1, .fptr = fizz_nif_mlirFloatAttrGetValueDouble, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFloatAttrDoubleGetChecked", .arity = 3, .fptr = fizz_nif_mlirFloatAttrDoubleGetChecked, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirFloatAttrDoubleGet", .arity = 3, .fptr = fizz_nif_mlirFloatAttrDoubleGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAFloat", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAFloat, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDictionaryAttrGetElementByName", .arity = 2, .fptr = fizz_nif_mlirDictionaryAttrGetElementByName, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDictionaryAttrGetElement", .arity = 2, .fptr = fizz_nif_mlirDictionaryAttrGetElement, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDictionaryAttrGetNumElements", .arity = 1, .fptr = fizz_nif_mlirDictionaryAttrGetNumElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDictionaryAttrGet", .arity = 3, .fptr = fizz_nif_mlirDictionaryAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsADictionary", .arity = 1, .fptr = fizz_nif_mlirAttributeIsADictionary, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirArrayAttrGetElement", .arity = 2, .fptr = fizz_nif_mlirArrayAttrGetElement, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirArrayAttrGetNumElements", .arity = 1, .fptr = fizz_nif_mlirArrayAttrGetNumElements, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirArrayAttrGet", .arity = 3, .fptr = fizz_nif_mlirArrayAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAArray", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAArray, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapAttrGetValue", .arity = 1, .fptr = fizz_nif_mlirAffineMapAttrGetValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapAttrGet", .arity = 1, .fptr = fizz_nif_mlirAffineMapAttrGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeIsAAffineMap", .arity = 1, .fptr = fizz_nif_mlirAttributeIsAAffineMap, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeGetNull", .arity = 0, .fptr = fizz_nif_mlirAttributeGetNull, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirExternalPassSignalFailure", .arity = 1, .fptr = fizz_nif_mlirExternalPassSignalFailure, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirCreateExternalPass", .arity = 9, .fptr = fizz_nif_mlirCreateExternalPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirParsePassPipeline", .arity = 2, .fptr = fizz_nif_mlirParsePassPipeline, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPrintPassPipeline", .arity = 3, .fptr = fizz_nif_mlirPrintPassPipeline, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpPassManagerAddOwnedPass", .arity = 2, .fptr = fizz_nif_mlirOpPassManagerAddOwnedPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPassManagerAddOwnedPass", .arity = 2, .fptr = fizz_nif_mlirPassManagerAddOwnedPass, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpPassManagerGetNestedUnder", .arity = 2, .fptr = fizz_nif_mlirOpPassManagerGetNestedUnder, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPassManagerGetNestedUnder", .arity = 2, .fptr = fizz_nif_mlirPassManagerGetNestedUnder, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPassManagerEnableVerifier", .arity = 2, .fptr = fizz_nif_mlirPassManagerEnableVerifier, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPassManagerEnableIRPrinting", .arity = 1, .fptr = fizz_nif_mlirPassManagerEnableIRPrinting, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPassManagerRun", .arity = 2, .fptr = fizz_nif_mlirPassManagerRun, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPassManagerGetAsOpPassManager", .arity = 1, .fptr = fizz_nif_mlirPassManagerGetAsOpPassManager, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPassManagerDestroy", .arity = 1, .fptr = fizz_nif_mlirPassManagerDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirPassManagerCreate", .arity = 1, .fptr = fizz_nif_mlirPassManagerCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAllPasses", .arity = 0, .fptr = fizz_nif_mlirRegisterAllPasses, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAllLLVMTranslations", .arity = 1, .fptr = fizz_nif_mlirRegisterAllLLVMTranslations, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegisterAllDialects", .arity = 1, .fptr = fizz_nif_mlirRegisterAllDialects, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectHandleLoadDialect", .arity = 2, .fptr = fizz_nif_mlirDialectHandleLoadDialect, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectHandleRegisterDialect", .arity = 2, .fptr = fizz_nif_mlirDialectHandleRegisterDialect, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectHandleInsertDialect", .arity = 2, .fptr = fizz_nif_mlirDialectHandleInsertDialect, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectHandleGetNamespace", .arity = 1, .fptr = fizz_nif_mlirDialectHandleGetNamespace, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapCompressUnusedSymbols", .arity = 4, .fptr = fizz_nif_mlirAffineMapCompressUnusedSymbols, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapReplace", .arity = 5, .fptr = fizz_nif_mlirAffineMapReplace, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetMinorSubMap", .arity = 2, .fptr = fizz_nif_mlirAffineMapGetMinorSubMap, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetMajorSubMap", .arity = 2, .fptr = fizz_nif_mlirAffineMapGetMajorSubMap, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetSubMap", .arity = 3, .fptr = fizz_nif_mlirAffineMapGetSubMap, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapIsPermutation", .arity = 1, .fptr = fizz_nif_mlirAffineMapIsPermutation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapIsProjectedPermutation", .arity = 1, .fptr = fizz_nif_mlirAffineMapIsProjectedPermutation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetNumInputs", .arity = 1, .fptr = fizz_nif_mlirAffineMapGetNumInputs, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetResult", .arity = 2, .fptr = fizz_nif_mlirAffineMapGetResult, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetNumResults", .arity = 1, .fptr = fizz_nif_mlirAffineMapGetNumResults, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetNumSymbols", .arity = 1, .fptr = fizz_nif_mlirAffineMapGetNumSymbols, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetNumDims", .arity = 1, .fptr = fizz_nif_mlirAffineMapGetNumDims, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetSingleConstantResult", .arity = 1, .fptr = fizz_nif_mlirAffineMapGetSingleConstantResult, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapIsSingleConstant", .arity = 1, .fptr = fizz_nif_mlirAffineMapIsSingleConstant, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapIsEmpty", .arity = 1, .fptr = fizz_nif_mlirAffineMapIsEmpty, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapIsMinorIdentity", .arity = 1, .fptr = fizz_nif_mlirAffineMapIsMinorIdentity, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapIsIdentity", .arity = 1, .fptr = fizz_nif_mlirAffineMapIsIdentity, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapPermutationGet", .arity = 3, .fptr = fizz_nif_mlirAffineMapPermutationGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapMinorIdentityGet", .arity = 3, .fptr = fizz_nif_mlirAffineMapMinorIdentityGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapMultiDimIdentityGet", .arity = 2, .fptr = fizz_nif_mlirAffineMapMultiDimIdentityGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapConstantGet", .arity = 2, .fptr = fizz_nif_mlirAffineMapConstantGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGet", .arity = 5, .fptr = fizz_nif_mlirAffineMapGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapZeroResultGet", .arity = 3, .fptr = fizz_nif_mlirAffineMapZeroResultGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapEmptyGet", .arity = 1, .fptr = fizz_nif_mlirAffineMapEmptyGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapDump", .arity = 1, .fptr = fizz_nif_mlirAffineMapDump, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapPrint", .arity = 3, .fptr = fizz_nif_mlirAffineMapPrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapEqual", .arity = 2, .fptr = fizz_nif_mlirAffineMapEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMapGetContext", .arity = 1, .fptr = fizz_nif_mlirAffineMapGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineBinaryOpExprGetRHS", .arity = 1, .fptr = fizz_nif_mlirAffineBinaryOpExprGetRHS, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineBinaryOpExprGetLHS", .arity = 1, .fptr = fizz_nif_mlirAffineBinaryOpExprGetLHS, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsABinary", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsABinary, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineCeilDivExprGet", .arity = 2, .fptr = fizz_nif_mlirAffineCeilDivExprGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsACeilDiv", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsACeilDiv, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineFloorDivExprGet", .arity = 2, .fptr = fizz_nif_mlirAffineFloorDivExprGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsAFloorDiv", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsAFloorDiv, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineModExprGet", .arity = 2, .fptr = fizz_nif_mlirAffineModExprGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsAMod", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsAMod, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineMulExprGet", .arity = 2, .fptr = fizz_nif_mlirAffineMulExprGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsAMul", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsAMul, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineAddExprGet", .arity = 2, .fptr = fizz_nif_mlirAffineAddExprGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsAAdd", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsAAdd, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineConstantExprGetValue", .arity = 1, .fptr = fizz_nif_mlirAffineConstantExprGetValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineConstantExprGet", .arity = 2, .fptr = fizz_nif_mlirAffineConstantExprGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsAConstant", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsAConstant, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineSymbolExprGetPosition", .arity = 1, .fptr = fizz_nif_mlirAffineSymbolExprGetPosition, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineSymbolExprGet", .arity = 2, .fptr = fizz_nif_mlirAffineSymbolExprGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsASymbol", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsASymbol, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineDimExprGetPosition", .arity = 1, .fptr = fizz_nif_mlirAffineDimExprGetPosition, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineDimExprGet", .arity = 2, .fptr = fizz_nif_mlirAffineDimExprGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsADim", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsADim, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprCompose", .arity = 2, .fptr = fizz_nif_mlirAffineExprCompose, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsFunctionOfDim", .arity = 2, .fptr = fizz_nif_mlirAffineExprIsFunctionOfDim, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsMultipleOf", .arity = 2, .fptr = fizz_nif_mlirAffineExprIsMultipleOf, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprGetLargestKnownDivisor", .arity = 1, .fptr = fizz_nif_mlirAffineExprGetLargestKnownDivisor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsPureAffine", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsPureAffine, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprIsSymbolicOrConstant", .arity = 1, .fptr = fizz_nif_mlirAffineExprIsSymbolicOrConstant, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprDump", .arity = 1, .fptr = fizz_nif_mlirAffineExprDump, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprPrint", .arity = 3, .fptr = fizz_nif_mlirAffineExprPrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprEqual", .arity = 2, .fptr = fizz_nif_mlirAffineExprEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAffineExprGetContext", .arity = 1, .fptr = fizz_nif_mlirAffineExprGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableWalkSymbolTables", .arity = 4, .fptr = fizz_nif_mlirSymbolTableWalkSymbolTables, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableReplaceAllSymbolUses", .arity = 3, .fptr = fizz_nif_mlirSymbolTableReplaceAllSymbolUses, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableErase", .arity = 2, .fptr = fizz_nif_mlirSymbolTableErase, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableInsert", .arity = 2, .fptr = fizz_nif_mlirSymbolTableInsert, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableLookup", .arity = 2, .fptr = fizz_nif_mlirSymbolTableLookup, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableDestroy", .arity = 1, .fptr = fizz_nif_mlirSymbolTableDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableCreate", .arity = 1, .fptr = fizz_nif_mlirSymbolTableCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableGetVisibilityAttributeName", .arity = 0, .fptr = fizz_nif_mlirSymbolTableGetVisibilityAttributeName, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirSymbolTableGetSymbolAttributeName", .arity = 0, .fptr = fizz_nif_mlirSymbolTableGetSymbolAttributeName, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIdentifierStr", .arity = 1, .fptr = fizz_nif_mlirIdentifierStr, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIdentifierEqual", .arity = 2, .fptr = fizz_nif_mlirIdentifierEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIdentifierGetContext", .arity = 1, .fptr = fizz_nif_mlirIdentifierGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirIdentifierGet", .arity = 2, .fptr = fizz_nif_mlirIdentifierGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirNamedAttributeGet", .arity = 2, .fptr = fizz_nif_mlirNamedAttributeGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeDump", .arity = 1, .fptr = fizz_nif_mlirAttributeDump, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributePrint", .arity = 3, .fptr = fizz_nif_mlirAttributePrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeEqual", .arity = 2, .fptr = fizz_nif_mlirAttributeEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeGetTypeID", .arity = 1, .fptr = fizz_nif_mlirAttributeGetTypeID, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeGetType", .arity = 1, .fptr = fizz_nif_mlirAttributeGetType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeGetContext", .arity = 1, .fptr = fizz_nif_mlirAttributeGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirAttributeParseGet", .arity = 2, .fptr = fizz_nif_mlirAttributeParseGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeDump", .arity = 1, .fptr = fizz_nif_mlirTypeDump, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypePrint", .arity = 3, .fptr = fizz_nif_mlirTypePrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeEqual", .arity = 2, .fptr = fizz_nif_mlirTypeEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeGetTypeID", .arity = 1, .fptr = fizz_nif_mlirTypeGetTypeID, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeGetContext", .arity = 1, .fptr = fizz_nif_mlirTypeGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeParseGet", .arity = 2, .fptr = fizz_nif_mlirTypeParseGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirValuePrint", .arity = 3, .fptr = fizz_nif_mlirValuePrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirValueDump", .arity = 1, .fptr = fizz_nif_mlirValueDump, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirValueGetType", .arity = 1, .fptr = fizz_nif_mlirValueGetType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpResultGetResultNumber", .arity = 1, .fptr = fizz_nif_mlirOpResultGetResultNumber, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpResultGetOwner", .arity = 1, .fptr = fizz_nif_mlirOpResultGetOwner, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockArgumentSetType", .arity = 2, .fptr = fizz_nif_mlirBlockArgumentSetType, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockArgumentGetArgNumber", .arity = 1, .fptr = fizz_nif_mlirBlockArgumentGetArgNumber, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockArgumentGetOwner", .arity = 1, .fptr = fizz_nif_mlirBlockArgumentGetOwner, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirValueIsAOpResult", .arity = 1, .fptr = fizz_nif_mlirValueIsAOpResult, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirValueIsABlockArgument", .arity = 1, .fptr = fizz_nif_mlirValueIsABlockArgument, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirValueEqual", .arity = 2, .fptr = fizz_nif_mlirValueEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockPrint", .arity = 3, .fptr = fizz_nif_mlirBlockPrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockGetArgument", .arity = 2, .fptr = fizz_nif_mlirBlockGetArgument, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockAddArgument", .arity = 3, .fptr = fizz_nif_mlirBlockAddArgument, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockGetNumArguments", .arity = 1, .fptr = fizz_nif_mlirBlockGetNumArguments, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockInsertOwnedOperationBefore", .arity = 3, .fptr = fizz_nif_mlirBlockInsertOwnedOperationBefore, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockInsertOwnedOperationAfter", .arity = 3, .fptr = fizz_nif_mlirBlockInsertOwnedOperationAfter, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockInsertOwnedOperation", .arity = 3, .fptr = fizz_nif_mlirBlockInsertOwnedOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockAppendOwnedOperation", .arity = 2, .fptr = fizz_nif_mlirBlockAppendOwnedOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockGetTerminator", .arity = 1, .fptr = fizz_nif_mlirBlockGetTerminator, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockGetFirstOperation", .arity = 1, .fptr = fizz_nif_mlirBlockGetFirstOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockGetNextInRegion", .arity = 1, .fptr = fizz_nif_mlirBlockGetNextInRegion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockGetParentRegion", .arity = 1, .fptr = fizz_nif_mlirBlockGetParentRegion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockGetParentOperation", .arity = 1, .fptr = fizz_nif_mlirBlockGetParentOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockEqual", .arity = 2, .fptr = fizz_nif_mlirBlockEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockDetach", .arity = 1, .fptr = fizz_nif_mlirBlockDetach, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockDestroy", .arity = 1, .fptr = fizz_nif_mlirBlockDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirBlockCreate", .arity = 3, .fptr = fizz_nif_mlirBlockCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionGetNextInOperation", .arity = 1, .fptr = fizz_nif_mlirRegionGetNextInOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetFirstRegion", .arity = 1, .fptr = fizz_nif_mlirOperationGetFirstRegion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionInsertOwnedBlockBefore", .arity = 3, .fptr = fizz_nif_mlirRegionInsertOwnedBlockBefore, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionInsertOwnedBlockAfter", .arity = 3, .fptr = fizz_nif_mlirRegionInsertOwnedBlockAfter, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionInsertOwnedBlock", .arity = 3, .fptr = fizz_nif_mlirRegionInsertOwnedBlock, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionAppendOwnedBlock", .arity = 2, .fptr = fizz_nif_mlirRegionAppendOwnedBlock, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionGetFirstBlock", .arity = 1, .fptr = fizz_nif_mlirRegionGetFirstBlock, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionEqual", .arity = 2, .fptr = fizz_nif_mlirRegionEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionDestroy", .arity = 1, .fptr = fizz_nif_mlirRegionDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirRegionCreate", .arity = 0, .fptr = fizz_nif_mlirRegionCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationMoveBefore", .arity = 2, .fptr = fizz_nif_mlirOperationMoveBefore, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationMoveAfter", .arity = 2, .fptr = fizz_nif_mlirOperationMoveAfter, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationVerify", .arity = 1, .fptr = fizz_nif_mlirOperationVerify, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationDump", .arity = 1, .fptr = fizz_nif_mlirOperationDump, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationPrintWithFlags", .arity = 4, .fptr = fizz_nif_mlirOperationPrintWithFlags, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationPrint", .arity = 3, .fptr = fizz_nif_mlirOperationPrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationRemoveAttributeByName", .arity = 2, .fptr = fizz_nif_mlirOperationRemoveAttributeByName, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationSetAttributeByName", .arity = 3, .fptr = fizz_nif_mlirOperationSetAttributeByName, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetAttributeByName", .arity = 2, .fptr = fizz_nif_mlirOperationGetAttributeByName, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetAttribute", .arity = 2, .fptr = fizz_nif_mlirOperationGetAttribute, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetNumAttributes", .arity = 1, .fptr = fizz_nif_mlirOperationGetNumAttributes, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetSuccessor", .arity = 2, .fptr = fizz_nif_mlirOperationGetSuccessor, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetNumSuccessors", .arity = 1, .fptr = fizz_nif_mlirOperationGetNumSuccessors, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetResult", .arity = 2, .fptr = fizz_nif_mlirOperationGetResult, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetNumResults", .arity = 1, .fptr = fizz_nif_mlirOperationGetNumResults, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationSetOperand", .arity = 3, .fptr = fizz_nif_mlirOperationSetOperand, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetOperand", .arity = 2, .fptr = fizz_nif_mlirOperationGetOperand, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetNumOperands", .arity = 1, .fptr = fizz_nif_mlirOperationGetNumOperands, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetNextInBlock", .arity = 1, .fptr = fizz_nif_mlirOperationGetNextInBlock, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetRegion", .arity = 2, .fptr = fizz_nif_mlirOperationGetRegion, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetNumRegions", .arity = 1, .fptr = fizz_nif_mlirOperationGetNumRegions, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetParentOperation", .arity = 1, .fptr = fizz_nif_mlirOperationGetParentOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetBlock", .arity = 1, .fptr = fizz_nif_mlirOperationGetBlock, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetName", .arity = 1, .fptr = fizz_nif_mlirOperationGetName, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetTypeID", .arity = 1, .fptr = fizz_nif_mlirOperationGetTypeID, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetLocation", .arity = 1, .fptr = fizz_nif_mlirOperationGetLocation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationGetContext", .arity = 1, .fptr = fizz_nif_mlirOperationGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationEqual", .arity = 2, .fptr = fizz_nif_mlirOperationEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationRemoveFromParent", .arity = 1, .fptr = fizz_nif_mlirOperationRemoveFromParent, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationDestroy", .arity = 1, .fptr = fizz_nif_mlirOperationDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationClone", .arity = 1, .fptr = fizz_nif_mlirOperationClone, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationCreate", .arity = 1, .fptr = fizz_nif_mlirOperationCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpPrintingFlagsUseLocalScope", .arity = 1, .fptr = fizz_nif_mlirOpPrintingFlagsUseLocalScope, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpPrintingFlagsPrintGenericOpForm", .arity = 1, .fptr = fizz_nif_mlirOpPrintingFlagsPrintGenericOpForm, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpPrintingFlagsEnableDebugInfo", .arity = 2, .fptr = fizz_nif_mlirOpPrintingFlagsEnableDebugInfo, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpPrintingFlagsElideLargeElementsAttrs", .arity = 2, .fptr = fizz_nif_mlirOpPrintingFlagsElideLargeElementsAttrs, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpPrintingFlagsDestroy", .arity = 1, .fptr = fizz_nif_mlirOpPrintingFlagsDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOpPrintingFlagsCreate", .arity = 0, .fptr = fizz_nif_mlirOpPrintingFlagsCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationStateEnableResultTypeInference", .arity = 1, .fptr = fizz_nif_mlirOperationStateEnableResultTypeInference, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationStateAddAttributes", .arity = 3, .fptr = fizz_nif_mlirOperationStateAddAttributes, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationStateAddSuccessors", .arity = 3, .fptr = fizz_nif_mlirOperationStateAddSuccessors, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationStateAddOwnedRegions", .arity = 3, .fptr = fizz_nif_mlirOperationStateAddOwnedRegions, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationStateAddOperands", .arity = 3, .fptr = fizz_nif_mlirOperationStateAddOperands, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationStateAddResults", .arity = 3, .fptr = fizz_nif_mlirOperationStateAddResults, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirOperationStateGet", .arity = 2, .fptr = fizz_nif_mlirOperationStateGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirModuleFromOperation", .arity = 1, .fptr = fizz_nif_mlirModuleFromOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirModuleGetOperation", .arity = 1, .fptr = fizz_nif_mlirModuleGetOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirModuleDestroy", .arity = 1, .fptr = fizz_nif_mlirModuleDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirModuleGetBody", .arity = 1, .fptr = fizz_nif_mlirModuleGetBody, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirModuleGetContext", .arity = 1, .fptr = fizz_nif_mlirModuleGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirModuleCreateParse", .arity = 2, .fptr = fizz_nif_mlirModuleCreateParse, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirModuleCreateEmpty", .arity = 1, .fptr = fizz_nif_mlirModuleCreateEmpty, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLocationPrint", .arity = 3, .fptr = fizz_nif_mlirLocationPrint, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLocationEqual", .arity = 2, .fptr = fizz_nif_mlirLocationEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLocationGetContext", .arity = 1, .fptr = fizz_nif_mlirLocationGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLocationUnknownGet", .arity = 1, .fptr = fizz_nif_mlirLocationUnknownGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLocationNameGet", .arity = 3, .fptr = fizz_nif_mlirLocationNameGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLocationFusedGet", .arity = 4, .fptr = fizz_nif_mlirLocationFusedGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLocationCallSiteGet", .arity = 2, .fptr = fizz_nif_mlirLocationCallSiteGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirLocationFileLineColGet", .arity = 4, .fptr = fizz_nif_mlirLocationFileLineColGet, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectRegistryDestroy", .arity = 1, .fptr = fizz_nif_mlirDialectRegistryDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectRegistryCreate", .arity = 0, .fptr = fizz_nif_mlirDialectRegistryCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectGetNamespace", .arity = 1, .fptr = fizz_nif_mlirDialectGetNamespace, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectEqual", .arity = 2, .fptr = fizz_nif_mlirDialectEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirDialectGetContext", .arity = 1, .fptr = fizz_nif_mlirDialectGetContext, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextIsRegisteredOperation", .arity = 2, .fptr = fizz_nif_mlirContextIsRegisteredOperation, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextEnableMultithreading", .arity = 2, .fptr = fizz_nif_mlirContextEnableMultithreading, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextGetOrLoadDialect", .arity = 2, .fptr = fizz_nif_mlirContextGetOrLoadDialect, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextGetNumLoadedDialects", .arity = 1, .fptr = fizz_nif_mlirContextGetNumLoadedDialects, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextAppendDialectRegistry", .arity = 2, .fptr = fizz_nif_mlirContextAppendDialectRegistry, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextGetNumRegisteredDialects", .arity = 1, .fptr = fizz_nif_mlirContextGetNumRegisteredDialects, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextGetAllowUnregisteredDialects", .arity = 1, .fptr = fizz_nif_mlirContextGetAllowUnregisteredDialects, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextSetAllowUnregisteredDialects", .arity = 2, .fptr = fizz_nif_mlirContextSetAllowUnregisteredDialects, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextDestroy", .arity = 1, .fptr = fizz_nif_mlirContextDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextEqual", .arity = 2, .fptr = fizz_nif_mlirContextEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirContextCreate", .arity = 0, .fptr = fizz_nif_mlirContextCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIDAllocatorAllocateTypeID", .arity = 1, .fptr = fizz_nif_mlirTypeIDAllocatorAllocateTypeID, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIDAllocatorDestroy", .arity = 1, .fptr = fizz_nif_mlirTypeIDAllocatorDestroy, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIDAllocatorCreate", .arity = 0, .fptr = fizz_nif_mlirTypeIDAllocatorCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIDHashValue", .arity = 1, .fptr = fizz_nif_mlirTypeIDHashValue, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIDEqual", .arity = 2, .fptr = fizz_nif_mlirTypeIDEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirTypeIDCreate", .arity = 1, .fptr = fizz_nif_mlirTypeIDCreate, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirStringRefEqual", .arity = 2, .fptr = fizz_nif_mlirStringRefEqual, .flags = 0},
  e.ErlNifFunc{.name = "fizz_nif_mlirStringRefCreateFromCString", .arity = 1, .fptr = fizz_nif_mlirStringRefCreateFromCString, .flags = 0},

};
