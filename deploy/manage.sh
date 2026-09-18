#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
action=${1:-status}
group=${2:-model}
if [[ ! -f .env ]]; then
  echo 'Missing .env; copy .env.example and configure it first.' >&2
  exit 1
fi
compose=(docker compose --project-directory "$repo_root" --env-file "$repo_root/.env" -f "$repo_root/deploy/vllm/compose.yaml")
case "$group" in
  model|all) services=(model) ;;
  *) echo 'Group must be model or all.' >&2; exit 2 ;;
esac
case "$action" in
  start) "${compose[@]}" up -d --build --no-recreate "${services[@]}" ;;
  start-loaded) "${compose[@]}" up -d --no-build --pull never --no-recreate "${services[@]}" ;;
  restart) "${compose[@]}" up -d --build --force-recreate "${services[@]}" ;;
  restart-loaded) "${compose[@]}" up -d --no-build --pull never --force-recreate "${services[@]}" ;;
  stop) "${compose[@]}" stop "${services[@]}" ;;
  status) "${compose[@]}" ps -a "${services[@]}" ;;
  logs) "${compose[@]}" logs --tail 100 -f "${services[@]}" ;;
  pull) "${compose[@]}" build --pull model ;;
  build) "${compose[@]}" build model ;;
  config) "${compose[@]}" config --quiet ;;
  check) python3 scripts/smoke_test.py ;;
  download) python3 scripts/download_model.py ;;
  verify) python3 scripts/download_model.py --verify-only ;;
  *) echo "Usage: $0 {start|start-loaded|restart|restart-loaded|stop|status|logs|pull|build|config|check|download|verify} [model|all]" >&2; exit 2 ;;
esac
