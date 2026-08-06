#!/usr/bin/env bash
# Builds Triton's core compiler libraries (dialects, passes, conversions) as a
# relocatable prebuilt for Beaver. Compiled on CI so the developer machine only
# downloads and links; no LLVM source build, no Python module, no GPU backend.
set -euo pipefail

triton_dir="${TRITON_SOURCE_DIR:?TRITON_SOURCE_DIR is required}"
llvm_syspath="${LLVM_SYSPATH:?LLVM_SYSPATH is required}"
output_dir="${TRITON_PREBUILT_OUTPUT_DIR:?TRITON_PREBUILT_OUTPUT_DIR is required}"
llvm_config="${llvm_syspath}/bin/llvm-config"
mlir_dir="$("${llvm_config}" --libdir)/cmake/mlir"
llvm_dir="$("${llvm_config}" --libdir)/cmake/llvm"
ccache="${TRITON_BUILD_WITH_CCACHE:-ON}"

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/triton-build.XXXXXX")"
cleanup() {
  rm -rf "${build_dir}"
}
trap cleanup EXIT

echo "Configuring Triton at ${triton_dir} against ${llvm_syspath}"
cmake -S "${triton_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_BUILD_PYTHON_MODULE=OFF \
  -DTRITON_BUILD_UT=OFF \
  -DTRITON_BUILD_PROTON=OFF \
  -DTRITON_BUILD_WITH_CCACHE="${ccache}" \
  -DTRITON_CACHE_PATH="${TRITON_CACHE_PATH:-${HOME}/.triton}" \
  -DLLVM_SYSPATH="${llvm_syspath}" \
  -DMLIR_DIR="${mlir_dir}" \
  -DLLVM_DIR="${llvm_dir}"

# Core dialect/pass/conversion libraries. These are OBJECT libraries in
# Triton; the archive step below bundles their objects into one library.
core_targets=(
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  TritonInstrumentIR
  GluonIR
  TritonTransforms
  TritonGPUTransforms
  TritonNvidiaGPUTransforms
  TritonInstrumentTransforms
  GluonTransforms
  TritonToTritonGPU
  TritonGPUToLLVM
  TritonInstrumentToLLVM
  TritonLLVMIR
  TritonAnalysis
  TritonTools
)

echo "Building ${#core_targets[@]} Triton core targets"
ninja -C "${build_dir}" "${core_targets[@]}"

mkdir -p "${output_dir}/lib" "${output_dir}/include"

echo "Bundling Triton object libraries"
mapfile -t objects < <(
  find "${build_dir}/lib" "${build_dir}/third_party" -path '*CMakeFiles/*.dir/*.o' 2>/dev/null
)
if ((${#objects[@]} == 0)); then
  echo "no Triton object files found under ${build_dir}" >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin)
    libtool -static -o "${output_dir}/lib/libtriton_core.a" "${objects[@]}"
    ;;
  Linux)
    ar rcs "${output_dir}/lib/libtriton_core.a" "${objects[@]}"
    ;;
  *)
    echo "unsupported platform for Triton prebuilt archive" >&2
    exit 1
    ;;
esac

echo "Copying Triton headers"
cp -R "${triton_dir}/include"/. "${output_dir}/include"/
cp -R "${build_dir}/include"/. "${output_dir}/include"/

echo "Triton prebuilt at ${output_dir}"
find "${output_dir}" -maxdepth 2 -type d | head -20
