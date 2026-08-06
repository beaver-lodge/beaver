#include "mlir/CAPI/Beaver.h"
#include "mlir-c/Debug.h"

#include <vector>

MLIR_CAPI_EXPORTED void beaverSetGlobalDebugTypes(const MlirStringRef *types,
                                                  intptr_t n) {
  // Convert MlirStringRef array to array of C strings
  std::vector<const char *> cstrings;
  cstrings.reserve(n);
  for (intptr_t i = 0; i < n; ++i) {
    cstrings.push_back(beaverStringRefGetData(types[i]));
  }

  // Call the underlying MLIR function
  mlirSetGlobalDebugTypes(cstrings.data(), n);
}
