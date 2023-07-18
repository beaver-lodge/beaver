add_mlir_public_c_api_library(
  MLIRCAPIElixir
  DISABLE_INSTALL
  Beaver.cpp
  Elixir.cpp
  LINK_LIBS
  PUBLIC
  MLIRElixir)
