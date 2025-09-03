# This lib is for dlopen or being linked by Zig/Rust

get_property(_capi_libraries GLOBAL PROPERTY MLIR_UPSTREAM_CAPI_LIBS)
add_mlir_aggregate(MLIRBeaver SHARED EMBED_LIBS MLIRCAPIElixir
                   ${_capi_libraries})
