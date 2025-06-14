#include "mlir/CAPI/Beaver.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "llvm/Support/ThreadPool.h"

using namespace mlir;

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetArgument(MlirPass pass) {
  auto argument = unwrap(pass)->getArgument();
  return wrap(argument);
}

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetName(MlirPass pass) {
  auto argument = unwrap(pass)->getName();
  return wrap(argument);
}

MLIR_CAPI_EXPORTED MlirStringRef beaverPassGetDescription(MlirPass pass) {
  return wrap(unwrap(pass)->getDescription());
}

MLIR_CAPI_EXPORTED MlirContext
beaverPassManagerGetContext(MlirPassManager passManager) {
  return wrap(unwrap(passManager)->getContext());
}

MLIR_CAPI_EXPORTED void
beaverPassManagerEnableTiming(MlirPassManager passManager) {
  unwrap(passManager)->enableTiming();
}

MLIR_CAPI_EXPORTED bool beaverIsOpNameTerminator(MlirStringRef op_name,
                                                 MlirContext context) {
  auto name = OperationName(unwrap(op_name), unwrap(context));
  return name.isRegistered() && name.mightHaveTrait<OpTrait::IsTerminator>();
}

MLIR_CAPI_EXPORTED void beaverContextGetOps(MlirContext context,
                                            MlirStringCallback insert,
                                            void *container) {
  for (const RegisteredOperationName &op :
       unwrap(context)->getRegisteredOperations()) {
    insert(wrap(op.getStringRef()), container);
  }
}

MLIR_CAPI_EXPORTED void beaverContextGetDialects(MlirContext context,
                                                 MlirStringCallback insert,
                                                 void *container) {
  for (auto dialect : unwrap(context)->getDialectRegistry().getDialectNames()) {
    insert(wrap(dialect), container);
  }
}

MLIR_CAPI_EXPORTED const char *
beaverStringRefGetData(MlirStringRef string_ref) {
  return string_ref.data;
}

MLIR_CAPI_EXPORTED size_t beaverStringRefGetLength(MlirStringRef string_ref) {
  return string_ref.length;
}

MLIR_CAPI_EXPORTED bool beaverIsNullContext(MlirContext w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullDialect(MlirDialect w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullDialectRegistry(MlirDialectRegistry w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverIsNullLocation(MlirLocation w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullModule(MlirModule w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullOperation(MlirOperation w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverIsNullRegion(MlirRegion w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullBlock(MlirBlock w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullValue(MlirValue w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullType(MlirType w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverIsNullAttribute(MlirAttribute w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverIsNullSymbolTable(MlirSymbolTable w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverIsNullExecutionEngine(MlirExecutionEngine w) {
  return !w.ptr;
}

MLIR_CAPI_EXPORTED MlirLocation
beaverOperationStateGetLocation(MlirOperationState state) {
  return state.location;
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumResults(MlirOperationState state) {
  return state.nResults;
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumOperands(MlirOperationState state) {
  return state.nOperands;
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumRegions(MlirOperationState state) {
  return state.nRegions;
}

MLIR_CAPI_EXPORTED intptr_t
beaverOperationStateGetNumAttributes(MlirOperationState state) {
  return state.nAttributes;
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverOperationStateGetName(MlirOperationState state) {
  return state.name;
}

MLIR_CAPI_EXPORTED MlirContext
beaverOperationStateGetContext(MlirOperationState state) {
  return mlirLocationGetContext(state.location);
}

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsSuccess(MlirLogicalResult res) {
  return mlirLogicalResultIsSuccess(res);
}

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsFailure(MlirLogicalResult res) {
  return mlirLogicalResultIsFailure(res);
}

MLIR_CAPI_EXPORTED
MlirIdentifier beaverNamedAttributeGetName(MlirNamedAttribute na) {
  return na.name;
}

MLIR_CAPI_EXPORTED
MlirAttribute beaverNamedAttributeGetAttribute(MlirNamedAttribute na) {
  return na.attribute;
}

MLIR_CAPI_EXPORTED MlirPass beaverPassCreate(
    void (*construct)(void *userData), void (*destruct)(void *userData),
    MlirLogicalResult (*initialize)(MlirContext ctx, void *userData),
    void *(*clone)(void *userData),
    void (*run)(MlirOperation op, MlirExternalPass pass, void *userData),
    MlirTypeID passID, MlirStringRef name, MlirStringRef argument,
    MlirStringRef description, MlirStringRef opName,
    intptr_t nDependentDialects, MlirDialectHandle *dependentDialects,
    void *userData) {
  return mlirCreateExternalPass(
      passID, name, argument, description, opName, nDependentDialects,
      dependentDialects,
      MlirExternalPassCallbacks{construct, destruct, initialize, clone, run},
      userData);
}

MLIR_CAPI_EXPORTED MlirAttribute beaverGetReassociationIndicesForReshape(
    MlirType sourceType, MlirType targetType) {
  auto indices = mlir::getReassociationIndicesForReshape(
      mlir::cast<RankedTensorType>(unwrap(sourceType)),
      mlir::cast<RankedTensorType>(unwrap(targetType)));
  OpBuilder b{unwrap(sourceType).getContext()};
  if (!indices) {
    return wrap(Attribute{});
  }
  return wrap(getReassociationIndicesAttribute(b, *indices));
}

MLIR_CAPI_EXPORTED void beaverLocationPrint(MlirLocation location,
                                            MlirStringCallback callback,
                                            void *userData) {
  if (auto loc = mlir::dyn_cast<FileLineColLoc>(unwrap(location))) {
    std::string s = loc.getFilename().str() + ":" +
                    std::to_string(loc.getLine()) + ":" +
                    std::to_string(loc.getColumn());
    callback(wrap(s), userData);
  } else {
    mlirLocationPrint(location, callback, userData);
  }
}

MLIR_CAPI_EXPORTED void mlirIdentifierPrint(MlirIdentifier identifier,
                                            MlirStringCallback callback,
                                            void *userData) {
  callback(mlirIdentifierStr(identifier), userData);
}

MLIR_CAPI_EXPORTED void beaverOperationPrintSpecializedFrom(
    MlirOperation op, MlirStringCallback callback, void *userData) {
  mlirOperationPrintWithFlags(
      op, wrap(&OpPrintingFlags().useLocalScope().printGenericOpForm(false)),
      callback, userData);
}

MLIR_CAPI_EXPORTED void
beaverOperationPrintGenericOpForm(MlirOperation op, MlirStringCallback callback,
                                  void *userData) {
  mlirOperationPrintWithFlags(
      op, wrap(&OpPrintingFlags().useLocalScope().printGenericOpForm(true)),
      callback, userData);
}

MLIR_CAPI_EXPORTED void beaverOperationDumpGeneric(MlirOperation op) {
  unwrap(op)->print(llvm::errs(),
                    OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs() << "\n";
}

template <typename T, typename EntityLookup, typename EntityGetter>
T getIRDLDefinedEntity(MlirStringRef dialect, MlirStringRef name,
                       MlirAttribute attrArr, EntityLookup lookup,
                       EntityGetter getter) {
  if (auto d =
          unwrap(attrArr).getContext()->getOrLoadDialect(unwrap(dialect))) {
    if (auto e = mlir::dyn_cast<ExtensibleDialect>(d)) {
      if (auto definition = lookup(e, unwrap(name))) {
        if (auto arr = mlir::dyn_cast<ArrayAttr>(unwrap(attrArr))) {
          return getter(definition, arr.getValue());
        }
      }
    }
  }
  return {};
}

MLIR_CAPI_EXPORTED MlirType beaverIRDLGetDefinedType(MlirStringRef dialect,
                                                     MlirStringRef type,
                                                     MlirAttribute params) {

  return wrap(getIRDLDefinedEntity<Type>(
      dialect, type, params,
      [](auto d, auto name) { return d->lookupTypeDefinition(name); },
      DynamicType::get));
}

MLIR_CAPI_EXPORTED MlirAttribute beaverIRDLGetDefinedAttr(
    MlirStringRef dialect, MlirStringRef attr, MlirAttribute params) {

  return wrap(getIRDLDefinedEntity<Attribute>(
      dialect, attr, params,
      [](auto d, auto name) { return d->lookupAttrDefinition(name); },
      DynamicAttr::get));
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverModuleApplyPatternsAndFoldGreedily(
    MlirModule module, MlirFrozenRewritePatternSet patterns) {
  return mlirApplyPatternsAndFoldGreedily(module, patterns, {});
}

MLIR_CAPI_EXPORTED bool beaverContextAddWork(MlirContext context,
                                             void (*task)(void *), void *arg) {
  if (unwrap(context)->isMultithreadingEnabled()) {
    unwrap(context)->getThreadPool().async([task, arg]() { task(arg); });
    return true;
  } else {
    return false;
  }
}

MLIR_CAPI_EXPORTED MlirType beaverDenseElementsAttrGetType(MlirAttribute attr) {
  return wrap(llvm::cast<DenseElementsAttr>(unwrap(attr)).getType());
}

MLIR_CAPI_EXPORTED intptr_t beaverShapedTypeGetNumElements(MlirType type) {
  return llvm::cast<ShapedType>(unwrap(type)).getNumElements();
}
