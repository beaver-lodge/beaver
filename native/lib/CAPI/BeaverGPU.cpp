#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"

using namespace mlir;

MLIR_CAPI_EXPORTED MlirStringRef beaverROCDLTargetAttrGetArch(MlirAttribute attribute) {
  auto target = cast<ROCDL::ROCDLTargetAttr>(unwrap(attribute));
  if (auto flags = target.getFlags())
    if (auto arch = flags.getAs<StringAttr>("arch")) return wrap(arch.getValue());
  return wrap(target.getChip());
}

MLIR_CAPI_EXPORTED unsigned beaverROCDLTargetAttrGetWaveSize(MlirAttribute attribute) {
  auto target = cast<ROCDL::ROCDLTargetAttr>(unwrap(attribute));
  if (auto flags = target.getFlags())
    if (auto wave = flags.getAs<IntegerAttr>("wavesize")) return wave.getInt();
  if (target.getFeatures().contains("+wavefrontsize32")) return 32;
  if (target.getFeatures().contains("+wavefrontsize64")) return 64;
  return 0;
}

MLIR_CAPI_EXPORTED MlirAttribute beaverROCDLTargetAttrConfigure(MlirAttribute attribute, MlirStringRef arch,
                                                              unsigned wavesize) {
  auto target = cast<ROCDL::ROCDLTargetAttr>(unwrap(attribute));
  auto *ctx = target.getContext();
  auto info = ROCDL::TargetInfo::get(unwrap(arch), wavesize, [&] { return emitError(UnknownLoc::get(ctx)); });
  if (failed(info) || info->isUnknown()) return {};
  auto id = ROCDL::TargetInfo::parseTargetID(unwrap(arch));
  unsigned resolvedWave = info->getWavefrontSize().value_or(0);
  SmallVector<StringRef> features;
  target.getFeatures().split(features, ',', -1, false);
  for (StringRef feature : features)
    if ((feature == "+wavefrontsize32" && resolvedWave != 32) ||
        (feature == "+wavefrontsize64" && resolvedWave != 64)) return {};
  std::string featureString = target.getFeatures().str();
  if (!featureString.empty()) featureString += ",";
  featureString += resolvedWave == 32 ? "+wavefrontsize32,-wavefrontsize64" : "+wavefrontsize64,-wavefrontsize32";
  NamedAttrList flags(target.getFlags());
  flags.set("arch", StringAttr::get(ctx, unwrap(arch)));
  flags.set("wavesize", IntegerAttr::get(IntegerType::get(ctx, 32), resolvedWave));
  flags.set(resolvedWave == 32 ? "no_wave64" : "wave64", UnitAttr::get(ctx));
  llvm::Triple triple(target.getTriple() == "amdgcn-amd-amdhsa" ? id->getTargetTripleString() : target.getTriple());
  triple.setArch(triple.getArch(), info->getSubArch());
  return wrap(ROCDL::ROCDLTargetAttr::get(ctx, target.getO(), triple.str(), info->getArchName(),
      featureString, target.getAbi(), flags.getDictionary(ctx), target.getLink()));
}

MLIR_CAPI_EXPORTED void beaverROCDLTargetApplyModuleFlags(MlirOperation module, MlirAttribute target) {
  auto info = ROCDL::TargetInfo::get(unwrap(beaverROCDLTargetAttrGetArch(target)),
      beaverROCDLTargetAttrGetWaveSize(target), [&] { return unwrap(module)->emitError(); });
  if (succeeded(info)) info->migrateArchFeaturesToModuleFlags(unwrap(module));
}

MLIR_CAPI_EXPORTED MlirStringRef beaverROCDLTargetAttrGetChip(MlirAttribute target) {
  return wrap(cast<ROCDL::ROCDLTargetAttr>(unwrap(target)).getChip());
}

MLIR_CAPI_EXPORTED MlirStringRef beaverGetNumWorkgroupAttributionsAttrName() {
  return wrap(llvm::StringRef("workgroup_attributions"));
}

MLIR_CAPI_EXPORTED MlirStringRef beaverGetContainerModuleAttrName() {
  return wrap(gpu::GPUDialect::getContainerModuleAttrName());
}

MLIR_CAPI_EXPORTED MlirAttribute beaverGPUObjectAttrGet(MlirAttribute target, int32_t format, MlirStringRef object) {
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

  return wrap(gpu::ObjectAttr::get(unwrap(target), compilationTarget,
                                   StringAttr::get(unwrap(target).getContext(), unwrap(object))));
}

MLIR_CAPI_EXPORTED bool beaverAttributeIsAGPUObject(MlirAttribute attribute) {
  return isa<gpu::ObjectAttr>(unwrap(attribute));
}

MLIR_CAPI_EXPORTED MlirStringRef beaverGPUObjectAttrGetObject(MlirAttribute attribute) {
  return wrap(cast<gpu::ObjectAttr>(unwrap(attribute)).getObject().getValue());
}
