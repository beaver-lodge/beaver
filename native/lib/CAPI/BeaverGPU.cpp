#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;

MLIR_CAPI_EXPORTED MlirStringRef beaverGetNumWorkgroupAttributionsAttrName() {
  return wrap(llvm::StringRef("workgroup_attributions"));
}

MLIR_CAPI_EXPORTED MlirStringRef beaverGetContainerModuleAttrName() {
  return wrap(gpu::GPUDialect::getContainerModuleAttrName());
}

MLIR_CAPI_EXPORTED MlirAttribute
beaverGPUObjectAttrGet(MlirAttribute target, int32_t format,
                       MlirStringRef object) {
  gpu::CompilationTarget compilationTarget;
  switch (format) {
  case 1:
    compilationTarget = gpu::CompilationTarget::Offload;
    break;
  case 2:
    compilationTarget = gpu::CompilationTarget::Assembly;
    break;
  case 3:
    compilationTarget = gpu::CompilationTarget::Binary;
    break;
  case 4:
    compilationTarget = gpu::CompilationTarget::Fatbin;
    break;
  default:
    return MlirAttribute{nullptr};
  }

  return wrap(gpu::ObjectAttr::get(
      unwrap(target), compilationTarget,
      StringAttr::get(unwrap(target).getContext(), unwrap(object))));
}

MLIR_CAPI_EXPORTED bool beaverAttributeIsAGPUObject(MlirAttribute attribute) {
  return isa<gpu::ObjectAttr>(unwrap(attribute));
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverGPUObjectAttrGetObject(MlirAttribute attribute) {
  return wrap(cast<gpu::ObjectAttr>(unwrap(attribute)).getObject().getValue());
}
