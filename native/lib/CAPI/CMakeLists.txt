if(DEFINED ENV{BEAVER_TRITON_PREBUILT_DIR})
  set(BEAVER_TRITON_PREBUILT_DIR "$ENV{BEAVER_TRITON_PREBUILT_DIR}")
endif()

add_mlir_public_c_api_library(
  MLIRCAPIBeaver
  DISABLE_INSTALL
  Beaver.cpp
  BeaverAnalysis.cpp
  BeaverExecution.cpp
  BeaverCompilerKernel.cpp
  BeaverIRDL.cpp
  BeaverDebug.cpp
  BeaverGPU.cpp
  BeaverLLVMIR.cpp
  BeaverActionTracing.cpp
  BeaverRemarks.cpp
  BeaverMemoryPlanner.cpp
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

if(BEAVER_TRITON_PREBUILT_DIR)
  file(READ "${BEAVER_TRITON_PREBUILT_DIR}/toolchain.json" _triton_metadata)
  string(JSON _triton_llvm_ref ERROR_VARIABLE _triton_ref_error GET "${_triton_metadata}" llvm_ref)
  list(GET LLVM_INCLUDE_DIRS 0 _llvm_include_dir)
  file(READ "${_llvm_include_dir}/llvm/Support/VCSRevision.h" _llvm_revision)
  string(LENGTH "${_triton_llvm_ref}" _triton_ref_length)
  string(FIND "${_llvm_revision}" "${_triton_llvm_ref}" _triton_ref_position)
  if(_triton_ref_error OR NOT _triton_ref_length EQUAL 40 OR _triton_ref_position EQUAL -1)
    message(FATAL_ERROR "Triton core must use the same LLVM revision; rebuild with mix beaver.build_triton")
  endif()
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
