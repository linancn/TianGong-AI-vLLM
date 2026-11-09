#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
base=$(python3 - "$root/deploy/deepseek-v41/runtime-manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))['base_image'])
PY
)
if ! docker image inspect "$base" >/dev/null 2>&1; then
  echo "Load or pull the pinned base_image from runtime-manifest.json first" >&2
  exit 1
fi
docker build --pull=false --build-arg "BASE_IMAGE=$base" \
  -t tiangong-vllm:deepseek-v41-v033-sm120-r2 "$root/deploy/deepseek-v41"
