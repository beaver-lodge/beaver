#!/usr/bin/env bash
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to resolve llvm/eudsl release assets" >&2
  exit 1
fi

install_dir="${1:-${LLVM_PREBUILT_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/llvm-prebuilt}}"
repo="${LLVM_EUDSL_REPO:-llvm/eudsl}"
tag="${LLVM_EUDSL_TAG:-llvm}"
asset_name="${LLVM_EUDSL_ASSET_NAME:-}"
asset_url="${LLVM_EUDSL_ASSET_URL:-}"
resolve_only="${LLVM_EUDSL_RESOLVE_ONLY:-0}"
token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if [[ -z "${install_dir}" || "${install_dir}" == "/" ]]; then
  echo "refusing to install LLVM into '${install_dir}'" >&2
  exit 1
fi

detect_platform() {
  local os_name arch_name uname_s uname_m
  uname_s="$(uname -s)"
  uname_m="$(uname -m)"

  case "${uname_s}" in
    Darwin)
      os_name="macos"
      ;;
    Linux)
      os_name="manylinux"
      ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
      os_name="windows"
      ;;
    *)
      echo "unsupported operating system: ${uname_s}" >&2
      exit 1
      ;;
  esac

  case "${uname_m}" in
    arm64|aarch64)
      if [[ "${os_name}" == "macos" ]]; then
        arch_name="arm64"
      else
        arch_name="aarch64"
      fi
      ;;
    x86_64|amd64)
      if [[ "${os_name}" == "windows" ]]; then
        arch_name="amd64"
      else
        arch_name="x86_64"
      fi
      ;;
    *)
      echo "unsupported architecture: ${uname_m}" >&2
      exit 1
      ;;
  esac

  printf '%s %s\n' "${os_name}" "${arch_name}"
}

if [[ -z "${asset_name}" ]]; then
  read -r asset_os asset_arch < <(detect_platform)
  asset_name="$(
    REPO="${repo}" \
    TAG="${tag}" \
    ASSET_OS="${asset_os}" \
    ASSET_ARCH="${asset_arch}" \
    GITHUB_TOKEN="${token}" \
    python3 - <<'PY'
import json
import os
import sys
import urllib.request


def make_request(url: str) -> urllib.request.Request:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


repo = os.environ["REPO"]
tag = os.environ["TAG"]
asset_os = os.environ["ASSET_OS"]
asset_arch = os.environ["ASSET_ARCH"]

release_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
with urllib.request.urlopen(make_request(release_url)) as response:
    release = json.load(response)

release_id = release["id"]
prefix = f"mlir_{asset_os}_{asset_arch}_"
matches = []

for page in range(1, 11):
    assets_url = (
        f"https://api.github.com/repos/{repo}/releases/{release_id}/assets"
        f"?per_page=100&page={page}"
    )
    with urllib.request.urlopen(make_request(assets_url)) as response:
        assets = json.load(response)
    if not assets:
        break
    for asset in assets:
        name = asset["name"]
        if name.startswith(prefix) and name.endswith(".tar.gz"):
            matches.append(asset)

if not matches:
    sys.stderr.write(
        f"no llvm/eudsl asset found for {asset_os}/{asset_arch} under tag {tag}\n"
    )
    sys.exit(1)

matches.sort(key=lambda asset: (asset.get("updated_at", ""), asset["name"]))
print(matches[-1]["name"])
PY
  )"
fi

if [[ -z "${asset_url}" ]]; then
  asset_url="https://github.com/${repo}/releases/download/${tag}/${asset_name}"
fi

if [[ "${resolve_only}" == "1" ]]; then
  printf 'LLVM_PREBUILT_ASSET_NAME=%s\n' "${asset_name}"
  printf 'LLVM_PREBUILT_URL=%s\n' "${asset_url}"
  exit 0
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/llvm-prebuilt.XXXXXX")"
archive_path="${tmp_root}/${asset_name}"
extract_dir="${tmp_root}/extract"

cleanup() {
  rm -rf "${tmp_root}"
}
trap cleanup EXIT

echo "Downloading ${asset_url}"
ASSET_URL="${asset_url}" ARCHIVE_PATH="${archive_path}" python3 - <<'PY'
import os
import urllib.request

urllib.request.urlretrieve(os.environ["ASSET_URL"], os.environ["ARCHIVE_PATH"])
PY

mkdir -p "${extract_dir}"
tar -xzf "${archive_path}" -C "${extract_dir}"

llvm_config="$(find "${extract_dir}" -path '*/bin/llvm-config' -type f | head -n 1)"
if [[ -z "${llvm_config}" ]]; then
  echo "could not locate llvm-config in ${asset_name}" >&2
  exit 1
fi

asset_root="$(cd "$(dirname "${llvm_config}")/.." && pwd)"
rm -rf "${install_dir}"
mkdir -p "${install_dir}"
cp -R "${asset_root}"/. "${install_dir}"/

llvm_config_path="${install_dir}/bin/llvm-config"
if [[ ! -x "${llvm_config_path}" ]]; then
  chmod +x "${llvm_config_path}" || true
fi

echo "Installed ${asset_name} into ${install_dir}"
echo "LLVM_CONFIG_PATH=${llvm_config_path}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    printf 'LLVM_CONFIG_PATH=%s\n' "${llvm_config_path}"
    printf 'LLVM_PREBUILT_DIR=%s\n' "${install_dir}"
    printf 'LLVM_PREBUILT_ASSET_NAME=%s\n' "${asset_name}"
    printf 'LLVM_PREBUILT_URL=%s\n' "${asset_url}"
  } >> "${GITHUB_ENV}"
fi
