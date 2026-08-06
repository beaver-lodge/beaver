add_mlir_public_c_api_library(
  MLIRCAPIBeaver
  DISABLE_INSTALL
  Beaver.cpp
  LINK_LIBS
  PUBLIC
  MLIRDebug)
