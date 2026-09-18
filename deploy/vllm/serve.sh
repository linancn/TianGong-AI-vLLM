#!/usr/bin/env bash
set -euo pipefail
args=(vllm serve "$@")
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  args+=(--speculative-config "$SPECULATIVE_CONFIG")
fi
exec "${args[@]}"
