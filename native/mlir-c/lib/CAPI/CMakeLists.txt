add_mlir_public_c_api_library(
  MLIRCAPIElixir
  DISABLE_INSTALL
  Beaver.cpp
  LINK_LIBS
  PUBLIC
  MLIRElixir
  MLIRRewrite)
