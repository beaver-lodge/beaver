# This lib is for dlopen or being linked by Zig/Rust
add_mlir_aggregate(
  MLIRBeaver
  SHARED
  EMBED_LIBS
  MLIRCAPIInterfaces
  MLIRCAPIAsync
  MLIRCAPIControlFlow
  MLIRCAPIConversion
  MLIRCAPIDebug
  MLIRCAPIExecutionEngine
  MLIRCAPIFunc
  MLIRCAPIGPU
  MLIRCAPIIR
  MLIRCAPIInterfaces
  MLIRCAPILLVM
  MLIRCAPILinalg
  MLIRCAPIPDL
  MLIRCAPIQuant
  MLIRCAPIRegisterEverything
  MLIRCAPISCF
  MLIRCAPIShape
  MLIRCAPISparseTensor
  MLIRCAPITensor
  MLIRCAPITransforms
  MLIRCAPIElixir)

set_target_properties(MLIRBeaver PROPERTIES NO_SONAME ON)
