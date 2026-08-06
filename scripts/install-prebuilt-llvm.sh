#!/usr/bin/env bash
# Thin compatibility wrapper around `mix beaver.install_prebuilt_llvm`.
#
# Prefer invoking the Mix task directly:
#
#   (cd scripts/install_llvm && mix beaver.install_prebuilt_llvm \
#     --install-dir priv/llvm-prebuilt)
#
# A leading positional argument is treated as the install directory, relative
# to the caller's working directory, matching the previous shell script.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install_dir="${1:-}"
if [[ -n "${install_dir}" && "${install_dir}" != /* ]]; then
  install_dir="$(pwd)/${install_dir}"
fi

cd "${script_dir}/install_llvm"

if [[ -n "${install_dir}" ]]; then
  exec mix beaver.install_prebuilt_llvm --install-dir "${install_dir}" "${@:2}"
else
  exec mix beaver.install_prebuilt_llvm
fi
