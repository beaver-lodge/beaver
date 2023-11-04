get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS Core Support AsmParser)

if(MLIR_INCLUDE_TESTS)
  set(test_libs
      MLIRTestStandardToLLVM
      MLIRAffineTransformsTestPasses
      MLIRDLTITestPasses
      MLIRGPUTestPasses
      MLIRLinalgTestPasses
      MLIRMathTestPasses
      MLIRSCFTestPasses
      MLIRShapeTestPasses
      MLIRSPIRVTestPasses
      MLIRStandardOpsTestPasses
      MLIRVectorTestPasses
      MLIRTestAnalysis
      MLIRTestDialect
      MLIRTestIR
      MLIRTestPass
      MLIRTestReducer
      MLIRTestRewrite
      MLIRTestTransforms)
endif()

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${test_libs}
    MLIRAnalysis
    MLIRDialect
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRIR)

message(STATUS "MLIR libs: ${LIBS}")

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
