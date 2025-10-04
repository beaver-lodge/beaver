#ifndef MLIR_C_BEAVER_CALLBACKTYPEDEF_H_
#define MLIR_C_BEAVER_CALLBACKTYPEDEF_H_

#include "mlir-c/AffineMap.h"
#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*MlirDiagnosticHandlerDeleteUserData)(void *);
typedef void *(*MlirExternalPassConstruct)(void *);
typedef void (*MlirExternalPassRun)(MlirOperation, MlirExternalPass, void *);
typedef MlirLogicalResult (*MlirGenericCallback)(MlirContext, void *);
typedef void (*MlirUnmanagedDenseResourceElementsAttrGetDeleteCallback)(
    void *, const void *, size_t, size_t);
typedef void (*MlirAffineMapCompressUnusedSymbolsPopulateResult)(void *,
                                                                 intptr_t,
                                                                 MlirAffineMap);
typedef void (*MlirSymbolTableWalkSymbolTablesCallback)(MlirOperation, bool,
                                                        void *);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_BEAVER_CALLBACKTYPEDEF_H_
