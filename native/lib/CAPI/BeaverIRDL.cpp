#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ExtensibleDialect.h"
using namespace mlir;
MLIR_CAPI_EXPORTED MlirAttribute beaverIRDLVariadicityArrayAttrGet(MlirContext context, intptr_t size,
                                                                   const int32_t *values) {
  if (size < 0 || (size && !values)) return wrap(Attribute{});
  SmallVector<irdl::VariadicityAttr> tags;
  for (intptr_t i = 0; i < size; ++i) {
    auto tag = irdl::symbolizeVariadicity(values[i]);
    if (!tag) return wrap(Attribute{});
    tags.push_back(irdl::VariadicityAttr::get(unwrap(context), *tag));
  }
  return wrap(irdl::VariadicityArrayAttr::get(unwrap(context), tags));
}
MLIR_CAPI_EXPORTED MlirAttribute beaverGetReassociationIndicesForReshape(MlirType sourceType, MlirType targetType) {
  auto indices = mlir::getReassociationIndicesForReshape(mlir::cast<RankedTensorType>(unwrap(sourceType)),
                                                         mlir::cast<RankedTensorType>(unwrap(targetType)));
  OpBuilder b{unwrap(sourceType).getContext()};
  if (!indices) return wrap(Attribute{});
  return wrap(getReassociationIndicesAttribute(b, *indices));
}
template <typename T, typename EntityLookup, typename EntityGetter>
T getIRDLDefinedEntity(MlirStringRef dialect, MlirStringRef name, MlirAttribute attrArr, EntityLookup lookup,
                       EntityGetter getter) {
  if (auto d = unwrap(attrArr).getContext()->getOrLoadDialect(unwrap(dialect))) {
    if (auto e = mlir::dyn_cast<ExtensibleDialect>(d)) {
      if (auto definition = lookup(e, unwrap(name))) {
        if (auto arr = mlir::dyn_cast<ArrayAttr>(unwrap(attrArr))) { return getter(definition, arr.getValue()); }
      }
    }
  }
  return {};
}
MLIR_CAPI_EXPORTED MlirType beaverIRDLGetDefinedType(MlirStringRef dialect, MlirStringRef type, MlirAttribute params) {
  return wrap(getIRDLDefinedEntity<Type>(
      dialect, type, params, [](auto d, auto name) { return d->lookupTypeDefinition(name); }, DynamicType::get));
}
MLIR_CAPI_EXPORTED MlirAttribute beaverIRDLGetDefinedAttr(MlirStringRef dialect, MlirStringRef attr,
                                                          MlirAttribute params) {
  return wrap(getIRDLDefinedEntity<Attribute>(
      dialect, attr, params, [](auto d, auto name) { return d->lookupAttrDefinition(name); }, DynamicAttr::get));
}
