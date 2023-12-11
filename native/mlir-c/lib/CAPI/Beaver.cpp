#include "mlir/CAPI/Beaver.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/ExtensibleDialect.h"

using namespace mlir;

MLIR_CAPI_EXPORTED MlirRewritePatternSet
beaverRewritePatternSetGet(MlirContext context) {
  return wrap(new RewritePatternSet(unwrap(context)));
}

MLIR_CAPI_EXPORTED MlirRewritePatternSet beaverPatternSetAddOwnedPDLPattern(
    MlirRewritePatternSet patternList, MlirModule module) {
  auto &set = unwrap(patternList)->add(PDLPatternModule(unwrap(module)));
  return wrap(&set);
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverApplyOwnedPatternSetOnRegion(
    MlirRegion region, MlirRewritePatternSet patternList) {
  return wrap(applyPatternsAndFoldGreedily(*unwrap(region),
                                           std::move(*unwrap(patternList))));
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverApplyOwnedPatternSetOnOperation(
    MlirOperation op, MlirRewritePatternSet patternList) {
  return wrap(applyPatternsAndFoldGreedily(unwrap(op),
                                           std::move(*unwrap(patternList))));
}

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

MLIR_CAPI_EXPORTED intptr_t
beaverGetNumRegisteredOperations(MlirContext context) {
  return unwrap(context)->getRegisteredOperations().size();
}

MLIR_CAPI_EXPORTED MlirRegisteredOperationName
beaverGetRegisteredOperationName(MlirContext context, intptr_t pos) {
  mlir::RegisteredOperationName name =
      unwrap(context)->getRegisteredOperations()[pos];
  return wrap(name);
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverRegisteredOperationNameGetDialectName(MlirRegisteredOperationName name) {
  return wrap(unwrap(name).getDialectNamespace());
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverRegisteredOperationNameGetOpName(MlirRegisteredOperationName name) {
  return wrap(unwrap(name).stripDialect());
}

MLIR_CAPI_EXPORTED void
beaverRegisteredOperationsOfDialect(MlirContext context, MlirStringRef dialect,
                                    MlirRegisteredOperationName *ret,
                                    size_t *num) {
  int i = 0;
  for (auto &op : unwrap(context)->getRegisteredOperations()) {
    if (std::string(op.getDialectNamespace()) == std::string(unwrap(dialect))) {
      if (i > 300) {
        llvm::errs() << "dialect " << unwrap(dialect) << " has more than 300 "
                     << "operations\n";
        exit(1);
      }
      ret[i] = wrap(op);
      i += 1;
    }
  }
  *num = i;
}

MLIR_CAPI_EXPORTED void
beaverRegisteredDialects(MlirContext context, MlirStringRef *ret, size_t *num) {
  int i = 0;
  for (auto dialect : unwrap(context)->getDialectRegistry().getDialectNames()) {
    if (i > 300) {
      llvm::errs() << "more than 300 dialect in registry" << dialect;
      exit(1);
    }
    ret[i] = wrap(dialect);
    i += 1;
  }
  *num = i;
}

MLIR_CAPI_EXPORTED void beaverEnterMultiThreadedExecution(MlirContext context) {
  unwrap(context)->enterMultiThreadedExecution();
}

MLIR_CAPI_EXPORTED void beaverExitMultiThreadedExecution(MlirContext context) {
  unwrap(context)->exitMultiThreadedExecution();
}

MLIR_CAPI_EXPORTED MlirValue beaverOperandGetValue(MlirOpOperand operand) {
  return wrap((*unwrap(operand)).get());
}

MLIR_CAPI_EXPORTED const char *
beaverStringRefGetData(MlirStringRef string_ref) {
  return string_ref.data;
}

MLIR_CAPI_EXPORTED size_t beaverStringRefGetLength(MlirStringRef string_ref) {
  return string_ref.length;
}

MLIR_CAPI_EXPORTED bool beaverContextIsNull(MlirContext w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverDialectIsNull(MlirDialect w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverDialectRegistryIsNull(MlirDialectRegistry w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverLocationIsNull(MlirLocation w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverModuleIsNull(MlirModule w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverOperationIsNull(MlirOperation w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverRegionIsNull(MlirRegion w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverBlockIsNull(MlirBlock w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverValueIsNull(MlirValue w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverTypeIsNull(MlirType w) { return !w.ptr; }
MLIR_CAPI_EXPORTED bool beaverAttributeIsNull(MlirAttribute w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverSymbolTableIsNull(MlirSymbolTable w) {
  return !w.ptr;
}
MLIR_CAPI_EXPORTED bool beaverMlirExecutionEngineIsNull(MlirExecutionEngine w) {
  return !w.ptr;
}

MLIR_CAPI_EXPORTED MlirLocation
beaverMlirOperationStateGetLocation(MlirOperationState state) {
  return state.location;
}

MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumResults(MlirOperationState state) {
  return state.nResults;
}

MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumOperands(MlirOperationState state) {
  return state.nOperands;
}

MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumRegions(MlirOperationState state) {
  return state.nRegions;
}

MLIR_CAPI_EXPORTED intptr_t
beaverMlirOperationStateGetNumAttributes(MlirOperationState state) {
  return state.nAttributes;
}

MLIR_CAPI_EXPORTED MlirStringRef
beaverMlirOperationStateGetName(MlirOperationState state) {
  return state.name;
}

MLIR_CAPI_EXPORTED MlirContext
beaverMlirOperationStateGetContext(MlirOperationState state) {
  return mlirLocationGetContext(state.location);
}

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsSuccess(MlirLogicalResult res) {
  return mlirLogicalResultIsSuccess(res);
}

MLIR_CAPI_EXPORTED bool beaverLogicalResultIsFailure(MlirLogicalResult res) {
  return mlirLogicalResultIsFailure(res);
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultSuccess() {
  return mlirLogicalResultSuccess();
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverLogicalResultFailure() {
  return mlirLogicalResultFailure();
}

MLIR_CAPI_EXPORTED MlirIdentifier beaverOperationGetName(MlirOperation op,
                                                         intptr_t pos) {
  return mlirOperationGetAttribute(op, pos).name;
}

MLIR_CAPI_EXPORTED MlirAttribute beaverOperationGetAttribute(MlirOperation op,
                                                             intptr_t pos) {
  return mlirOperationGetAttribute(op, pos).attribute;
}

MLIR_CAPI_EXPORTED
MlirIdentifier beaverMlirNamedAttributeGetName(MlirNamedAttribute na) {
  return na.name;
}

MLIR_CAPI_EXPORTED
MlirAttribute beaverMlirNamedAttributeGetAttribute(MlirNamedAttribute na) {
  return na.attribute;
}

MLIR_CAPI_EXPORTED MlirPass beaverCreateExternalPass(
    void (*construct)(void *userData), MlirTypeID passID, MlirStringRef name,
    MlirStringRef argument, MlirStringRef description, MlirStringRef opName,
    intptr_t nDependentDialects, MlirDialectHandle *dependentDialects,
    void (*destruct)(void *userData),
    MlirLogicalResult (*initialize)(MlirContext ctx, void *userData),
    void *(*clone)(void *userData),
    void (*run)(MlirOperation op, MlirExternalPass pass, void *userData),
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
      unwrap(sourceType).cast<RankedTensorType>(),
      unwrap(targetType).cast<RankedTensorType>());
  OpBuilder b{unwrap(sourceType).getContext()};
  if (!indices) {
    return wrap(Attribute{});
  }
  return wrap(getReassociationIndicesAttribute(b, *indices));
}

MLIR_CAPI_EXPORTED void beaverLocationPrint(MlirLocation location,
                                            MlirStringCallback callback,
                                            void *userData) {
  if (auto loc = unwrap(location)->dyn_cast<FileLineColLoc>()) {
    std::string s = loc.getFilename().str() + ":" +
                    std::to_string(loc.getLine()) + ":" +
                    std::to_string(loc.getColumn());
    callback(wrap(s), userData);
  } else {
    mlirLocationPrint(location, callback, userData);
  }
}

MLIR_CAPI_EXPORTED void beaverOperationDumpGeneric(MlirOperation op) {
  unwrap(op)->print(llvm::errs(),
                    OpPrintingFlags().useLocalScope().printGenericOpForm());
  llvm::errs() << "\n";
}

MLIR_CAPI_EXPORTED MlirLogicalResult beaverLoadIRDLDialects(MlirModule module) {
  return wrap(irdl::loadDialects(unwrap(module)));
}

template <typename T, typename EntityLookup, typename EntityGetter>
T getIRDLDefinedEntity(MlirStringRef dialect, MlirStringRef name,
                       MlirAttribute attrArr, EntityLookup lookup,
                       EntityGetter getter) {
  if (auto d =
          unwrap(attrArr).getContext()->getOrLoadDialect(unwrap(dialect))) {
    if (auto e = llvm::dyn_cast<ExtensibleDialect>(d)) {
      if (auto definition = lookup(e, unwrap(name))) {
        if (auto arr = unwrap(attrArr).dyn_cast<ArrayAttr>()) {
          return getter(definition, arr.getValue());
        }
      }
    }
  }
  return {};
}

MLIR_CAPI_EXPORTED MlirType beaverGetIRDLDefinedType(MlirStringRef dialect,
                                                     MlirStringRef type,
                                                     MlirAttribute params) {

  return wrap(getIRDLDefinedEntity<Type>(
      dialect, type, params,
      [](auto d, auto name) { return d->lookupTypeDefinition(name); },
      DynamicType::get));
}

MLIR_CAPI_EXPORTED MlirAttribute beaverGetIRDLDefinedAttr(
    MlirStringRef dialect, MlirStringRef attr, MlirAttribute params) {

  return wrap(getIRDLDefinedEntity<Attribute>(
      dialect, attr, params,
      [](auto d, auto name) { return d->lookupAttrDefinition(name); },
      DynamicAttr::get));
}
