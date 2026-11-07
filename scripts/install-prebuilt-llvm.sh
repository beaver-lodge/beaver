#!/usr/bin/env bash
# Thin compatibility wrapper around `mix beaver.install_prebuilt_llvm`.
#
# Prefer invoking the Mix task directly:
#
#   (cd scripts/install_llvm && mix deps.get && mix beaver.install_prebuilt_llvm \
#     --install-dir priv/llvm-prebuilt)
#
# A leading positional argument is treated as the install directory, relative
# to the caller's working directory, matching the previous shell script.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 && "$1" != -* ]]; then
  install_dir="$1"
  if [[ "${install_dir}" != /* ]]; then
    install_dir="${PWD}/${install_dir}"
  fi
  shift
  set -- --install-dir "${install_dir}" "$@"
fi

cd "${script_dir}/install_llvm"
mix deps.get

exec mix beaver.install_prebuilt_llvm "$@"
