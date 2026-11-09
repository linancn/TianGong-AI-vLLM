#!/usr/bin/env bash
# CPU-only build: requires a local checkout of the immutable publishing tools.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: build.sh <embedding-model-checkout> <linux-x86_64-uv-0.12.16> [image-tag]' >&2
  exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
tools_sha=2d0c7d36a00b14dc1cc19d7ac6f94887f9d45d35
uv_binary=$(realpath "$2")
[[ "$("$uv_binary" --version)" == 'uv 0.12.16'* ]] || { echo 'Expected uv 0.12.16' >&2; exit 1; }
git -C "$1" cat-file -e "$tools_sha^{commit}"
mkdir -p "$root/output"
context=$(mktemp -d "$root/output/lca-build.XXXXXX")
trap 'rm -rf -- "$context"' EXIT
mkdir "$context/tools"
git -C "$1" archive "$tools_sha" | tar -xf - -C "$context/tools"
cp -- "$uv_binary" "$context/uv"
cp -- "$root/deploy/lca-nemotron/"{Dockerfile,server.py,.dockerignore} "$context/"
docker build --pull=false -t "${3:-tiangong-lca-nemotron:st5.3.0-r2}" "$context"
