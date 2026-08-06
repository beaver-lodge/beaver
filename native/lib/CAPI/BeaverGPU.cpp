#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

MLIR_CAPI_EXPORTED MlirStringRef beaverGetNumWorkgroupAttributionsAttrName() {
  return wrap(llvm::StringRef("workgroup_attributions"));
}

MLIR_CAPI_EXPORTED MlirStringRef beaverGetContainerModuleAttrName() {
  return wrap(gpu::GPUDialect::getContainerModuleAttrName());
}
