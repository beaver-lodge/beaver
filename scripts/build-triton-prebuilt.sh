#!/usr/bin/env bash
# Compatibility entrypoint; the standalone package owns the native build recipe.
set -euo pipefail
for variable in TRITON_SOURCE_DIR LLVM_SYSPATH TRITON_PREBUILT_OUTPUT_DIR TRITON_CACHE_PATH; do
  value="${!variable:-}"
  if [[ -n "$value" && "$value" != /* ]]; then
    printf -v "$variable" '%s/%s' "$PWD" "$value"
    export "$variable"
  fi
done
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/install_llvm"
mix deps.get
exec mix openai_triton.build "$@"
