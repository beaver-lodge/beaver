if(DEFINED ENV{BEAVER_TRITON_PREBUILT_DIR})
  set(BEAVER_TRITON_PREBUILT_DIR "$ENV{BEAVER_TRITON_PREBUILT_DIR}")
endif()

include(CheckCXXSourceCompiles)
set(_beaver_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(_beaver_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_REQUIRED_INCLUDES "${MLIR_INCLUDE_DIRS}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
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
set(CMAKE_REQUIRED_INCLUDES "${_beaver_saved_required_includes}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_beaver_saved_try_compile_target_type}")

add_mlir_public_c_api_library(
  MLIRCAPIBeaver
  DISABLE_INSTALL
  Beaver.cpp
  BeaverIRDL.cpp
  BeaverDebug.cpp
  BeaverGPU.cpp
  BeaverLLVMIR.cpp
  BeaverActionTracing.cpp
  Triton.cpp
  LINK_LIBS
  PUBLIC
  MLIRDebug
  LLVMSupport
  LLVMCore
  LLVMIRReader
  LLVMTarget
  LLVMCodeGen
  LLVMNVPTXCodeGen)

if(BEAVER_HAS_MLIR_CONTEXT_TRANSIENT_SCOPE)
  foreach(_target MLIRCAPIBeaver obj.MLIRCAPIBeaver)
    target_compile_definitions(${_target}
      PUBLIC
      BEAVER_HAS_MLIR_CONTEXT_TRANSIENT_SCOPE=1)
  endforeach()
endif()

if(BEAVER_HAS_PACKED_TRANSFORM_PARAMS)
  foreach(_target MLIRCAPIBeaver obj.MLIRCAPIBeaver)
    target_compile_definitions(${_target}
      PUBLIC
      BEAVER_HAS_PACKED_TRANSFORM_PARAMS=1)
  endforeach()
endif()

if(BEAVER_TRITON_PREBUILT_DIR)
  foreach(_target MLIRCAPIBeaver obj.MLIRCAPIBeaver)
    target_include_directories(${_target}
      PUBLIC
      "${BEAVER_TRITON_PREBUILT_DIR}/include"
      "${BEAVER_TRITON_PREBUILT_DIR}/include/third_party")
    target_compile_definitions(${_target}
      PUBLIC
      BEAVER_ENABLE_TRITON=1)
  endforeach()
  target_link_libraries(MLIRCAPIBeaver
    PUBLIC
    "${BEAVER_TRITON_PREBUILT_DIR}/lib/libtriton_core.a")
endif()
