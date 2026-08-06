if(DEFINED ENV{BEAVER_TRITON_PREBUILT_DIR})
  set(BEAVER_TRITON_PREBUILT_DIR "$ENV{BEAVER_TRITON_PREBUILT_DIR}")
endif()

add_mlir_public_c_api_library(
  MLIRCAPIBeaver
  DISABLE_INSTALL
  Beaver.cpp
  Triton.cpp
  LINK_LIBS
  PUBLIC
  MLIRDebug)

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
