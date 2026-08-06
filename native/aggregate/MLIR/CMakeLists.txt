# This lib is for dlopen or being linked by Zig/Rust

get_property(_capi_libraries GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)

if(DEFINED ENV{BEAVER_TRITON_PREBUILT_DIR})
  set(BEAVER_TRITON_PREBUILT_DIR "$ENV{BEAVER_TRITON_PREBUILT_DIR}")
  set(_triton_core_lib "${BEAVER_TRITON_PREBUILT_DIR}/lib/libtriton_core.a")
endif()

add_mlir_aggregate(MLIRBeaver SHARED
                   PUBLIC_LIBS ${_triton_core_lib}
                   EMBED_LIBS MLIRCAPIElixir
                   ${_capi_libraries})
