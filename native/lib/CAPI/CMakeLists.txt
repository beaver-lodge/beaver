if(DEFINED ENV{BEAVER_TRITON_PREBUILT_DIR})
  set(BEAVER_TRITON_PREBUILT_DIR "$ENV{BEAVER_TRITON_PREBUILT_DIR}")
endif()

include(CheckCXXSourceCompiles)
set(_beaver_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(_beaver_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_REQUIRED_INCLUDES "${MLIR_INCLUDE_DIRS}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_source_compiles(
  "#include <mlir/IR/MLIRContext.h>
   int main() {
     auto enable = &mlir::MLIRContext::enableThreadSafeLocking;
     auto query = &mlir::MLIRContext::isThreadSafeLockingEnabled;
     (void)enable; (void)query;
   }"
  BEAVER_HAS_MLIR_CONTEXT_LOCKING)
check_cxx_source_compiles(
  "#include <mlir/Pass/PassInstrumentation.h>
   int main() { auto query = &mlir::PassInstrumentation::isThreadSafe; (void)query; }"
  BEAVER_HAS_MLIR_THREAD_SAFE_PASS_INSTRUMENTATION)
check_cxx_source_compiles(
  "#include <mlir/Dialect/LLVMIR/ROCDLTargetInfo.h>
   #include <mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h>
   int main() { mlir::ConvertGpuOpsToROCDLOpsOptions options; options.arch = \"gfx1100\"; options.waveSize = 32; }"
  BEAVER_HAS_ROCDL_TARGET_INFO)
check_cxx_source_compiles(
  "#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
   int main() { mlir::bufferization::StaticMemoryPlannerAnalysisPassOptions options; (void)options; }"
  BEAVER_HAS_STATIC_MEMORY_PLANNER)
# Older LLVMs expose the planner API but mishandle skipped allocations and
# shared deallocs. Probe the installed tool before advertising this capability.
if(BEAVER_HAS_STATIC_MEMORY_PLANNER)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/planner-probe.mlir" [=[
func.func @probe(%n: index, %c: i1) {
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc(%n) : memref<?xf32>
  %ac = memref.cast %a : memref<4xf32> to memref<?xf32>
  %s = arith.select %c, %ac, %b : memref<?xf32>
  memref.dealloc %s : memref<?xf32>
  %x = memref.alloc() : memref<4xcomplex<f32>>
  memref.dealloc %x : memref<4xcomplex<f32>>
  %y = memref.alloc() : memref<4xf32, strided<[2]>>
  memref.dealloc %y : memref<4xf32, strided<[2]>>
  return
}
]=])
  execute_process(COMMAND "${LLVM_TOOLS_BINARY_DIR}/mlir-opt${CMAKE_EXECUTABLE_SUFFIX}"
    "${CMAKE_CURRENT_BINARY_DIR}/planner-probe.mlir"
    "--pass-pipeline=builtin.module(func.func(static-memory-planner-analysis))"
    RESULT_VARIABLE _planner_result OUTPUT_VARIABLE _planner_output ERROR_QUIET TIMEOUT 15)
  if(NOT "${_planner_result}" STREQUAL "0" OR _planner_output MATCHES "memref.view")
    set(BEAVER_HAS_STATIC_MEMORY_PLANNER OFF)
    message(STATUS "Static memory planner disabled: LLVM lacks skipped-allocation correctness fixes")
  endif()
endif()
check_cxx_source_compiles(
  "#include <mlir-c/IR.h>
   int main() {
     auto *begin = &mlirContextBeginTransientScope;
     auto *end = &mlirContextEndTransientScope;
     auto *active = &mlirContextIsInTransientScope;
     (void)begin; (void)end; (void)active;
     return 0;
   }"
  BEAVER_HAS_MLIR_CONTEXT_TRANSIENT_SCOPE)
check_cxx_source_compiles(
  "#include <mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h>
   int main() {
     auto getter = &mlir::transform::TileUsingForOp::getPackedInterchange;
     (void)getter;
     return 0;
   }"
  BEAVER_HAS_PACKED_TRANSFORM_PARAMS)
check_cxx_source_compiles(
  "#include <mlir/IR/SymbolTable.h>
   int main() {
     auto name = mlir::SymbolOpInterface::getDefaultVisibilityAttrName();
     (void)name;
     return 0;
   }"
  BEAVER_HAS_MLIR_DEFAULT_SYMBOL_VISIBILITY_ATTRIBUTE_NAME)
check_cxx_source_compiles(
  "#include <mlir/IR/Operation.h>
   int main() {
     mlir::Operation *operation = nullptr;
     operation->getName().walkInherentAttrs(
       operation,
       [](llvm::StringRef, mlir::Attribute &) {});
     return 0;
   }"
  BEAVER_HAS_MLIR_INHERENT_ATTRIBUTE_VISITOR)
check_cxx_source_compiles(
  "#include <mlir/Transforms/CompositePass.h>
   #include <mlir/Transforms/Passes.h>
   int main() {
     auto pass = mlir::createCompositeFixedPointPass(
       \"beaver-probe\",
       [](mlir::OpPassManager &) {},
       1,
       mlir::ConvergenceFailureAction::Error);
     return pass ? 0 : 1;
   }"
  BEAVER_HAS_MLIR_COMPOSITE_FAILURE_ACTION)
set(CMAKE_REQUIRED_INCLUDES "${_beaver_saved_required_includes}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_beaver_saved_try_compile_target_type}")

add_mlir_public_c_api_library(
  MLIRCAPIBeaver
  DISABLE_INSTALL
  Beaver.cpp
  BeaverExecution.cpp
  BeaverCompilerKernel.cpp
  BeaverIRDL.cpp
  BeaverDebug.cpp
  BeaverGPU.cpp
  BeaverLLVMIR.cpp
  BeaverActionTracing.cpp
  BeaverRemarks.cpp
  BeaverMemoryPlanner.cpp
  # Upstream d1ec07e04314 fixes DataLayout handling after the pinned prebuilt.
  # Defining its complete translation unit here prevents the older static
  # archive member from being pulled in. Remove on the next fixed LLVM pin.
  IndexToLLVM.cpp
  Triton.cpp
  LINK_LIBS
  PUBLIC
  MLIRDebug
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  LLVMSupport
  LLVMCore
  LLVMIRReader
  LLVMTarget
  LLVMCodeGen
  LLVMNVPTXCodeGen)

# External providers may be built with C++ exceptions enabled. Keep the
# exception barrier local to the compiler-kernel boundary while the rest of
# MLIR/Beaver remains compiled with LLVM's normal -fno-exceptions policy.
if(MSVC)
  set_source_files_properties(BeaverCompilerKernel.cpp
    PROPERTIES COMPILE_OPTIONS "/EHsc")
else()
  set_source_files_properties(BeaverCompilerKernel.cpp
    PROPERTIES COMPILE_OPTIONS "-fexceptions")
endif()

foreach(_feature MLIR_CONTEXT_LOCKING MLIR_THREAD_SAFE_PASS_INSTRUMENTATION ROCDL_TARGET_INFO STATIC_MEMORY_PLANNER MLIR_CONTEXT_TRANSIENT_SCOPE PACKED_TRANSFORM_PARAMS
    MLIR_DEFAULT_SYMBOL_VISIBILITY_ATTRIBUTE_NAME MLIR_INHERENT_ATTRIBUTE_VISITOR MLIR_COMPOSITE_FAILURE_ACTION)
  if(BEAVER_HAS_${_feature})
    foreach(_target MLIRCAPIBeaver obj.MLIRCAPIBeaver)
      target_compile_definitions(${_target} PUBLIC BEAVER_HAS_${_feature}=1)
    endforeach()
  endif()
endforeach()

if(BEAVER_TRITON_PREBUILT_DIR)
  foreach(_target MLIRCAPIBeaver obj.MLIRCAPIBeaver)
    target_include_directories(${_target}
      PUBLIC
      "${BEAVER_TRITON_PREBUILT_DIR}/include"
      "${BEAVER_TRITON_PREBUILT_DIR}/include/third_party")
    # Triton's public headers require C++20 concepts.
    target_compile_features(${_target} PUBLIC cxx_std_20)
    target_compile_definitions(${_target} PUBLIC BEAVER_ENABLE_TRITON=1)
  endforeach()
  target_link_libraries(MLIRCAPIBeaver
    PUBLIC
    "${BEAVER_TRITON_PREBUILT_DIR}/lib/libtriton_core.a")
endif()
