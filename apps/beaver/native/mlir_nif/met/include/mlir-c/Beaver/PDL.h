#ifndef APPS_MLIR_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PDL_H_
#define APPS_MLIR_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PDL_H_

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirPDLPatternModule, void);
DEFINE_C_API_STRUCT(MlirRewritePatternSet, void);

#undef DEFINE_C_API_STRUCT

MLIR_CAPI_EXPORTED MlirPDLPatternModule beaverPDLPatternGet(MlirModule module);

MLIR_CAPI_EXPORTED MlirRewritePatternSet
beaverRewritePatternSetGet(MlirContext context);

MLIR_CAPI_EXPORTED MlirRewritePatternSet beaverPatternSetAddOwnedPDLPattern(
    MlirRewritePatternSet patternList, MlirPDLPatternModule pdlPattern);

MLIR_CAPI_EXPORTED MlirLogicalResult beaverApplyOwnedPatternSetOnRegion(
    MlirRegion region, MlirRewritePatternSet patternList);

MLIR_CAPI_EXPORTED MlirLogicalResult beaverApplyOwnedPatternSetOnOperation(
    MlirOperation op, MlirRewritePatternSet patternList);

#ifdef __cplusplus
}
#endif

#endif // APPS_MLIR_NATIVE_MLIR_NIF_MET_INCLUDE_MLIR_C_BEAVER_PDL_H_
