#!/usr/bin/env bash
set -xe

# manylinux prep
if [[ -f "/etc/centos-release" ]]; then
  for i in $(seq 1 5); do
    dnf install -y elixir zig && s=0 && break || s=$? && sleep 15
  done

elif [[ -f "/etc/alpine-release" ]]; then
  # musllinux prep
  # ccache already present
  apk add elixir zig
fi
