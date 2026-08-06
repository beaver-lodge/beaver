if(DEFINED ENV{BEAVER_TRITON_PREBUILT_DIR})
  set(BEAVER_TRITON_PREBUILT_DIR "$ENV{BEAVER_TRITON_PREBUILT_DIR}")
endif()

add_mlir_public_c_api_library(
  MLIRCAPIElixir
  DISABLE_INSTALL
  Beaver.cpp
  Elixir.cpp
  Triton.cpp
  LINK_LIBS
  PUBLIC
  MLIRElixir
  MLIRDebug)

if(BEAVER_TRITON_PREBUILT_DIR)
  target_include_directories(MLIRCAPIElixir
    PUBLIC
    "${BEAVER_TRITON_PREBUILT_DIR}/include")
  target_compile_definitions(MLIRCAPIElixir
    PUBLIC
    BEAVER_ENABLE_TRITON=1)
  target_link_libraries(MLIRCAPIElixir
    PUBLIC
    "${BEAVER_TRITON_PREBUILT_DIR}/lib/libtriton_core.a")
endif()
