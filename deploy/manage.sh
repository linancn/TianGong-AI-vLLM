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
case "$group" in
  model|all)
    compose=(docker compose --project-directory "$repo_root" --env-file "$repo_root/.env" -f "$repo_root/deploy/vllm/compose.yaml")
    service=model
    ;;
  embed)
    compose=(docker compose --project-name tiangong-embed --project-directory "$repo_root" --env-file "$repo_root/.env" -f "$repo_root/deploy/embed/compose.yaml")
    service=embed
    ;;
  embed3)
    compose=(docker compose --project-name tiangong-embed3 --project-directory "$repo_root" --env-file "$repo_root/.env" -f "$repo_root/deploy/embed3/compose.yaml")
    service=embed
    ;;
  *) echo 'Group must be model, embed, embed3, or all (model alias).' >&2; exit 2 ;;
esac
if [[ "$group" == embed3 && -f "$repo_root/output/instances/embed3.compose.yaml" ]]; then
  compose+=(-f "$repo_root/output/instances/embed3.compose.yaml")
fi
download_model=$service
if [[ "$group" == embed3 ]]; then
  download_model=embed3
fi
check_embed_conflict() {
  local other_project running
  case "$group" in
    embed) other_project=tiangong-embed3 ;;
    embed3) other_project=tiangong-embed ;;
    *) return ;;
  esac
  running=$(docker ps --quiet --filter "label=com.docker.compose.project=$other_project")
  if [[ -n "$running" ]]; then
    echo "Cannot start $group while $other_project is running on this host; stop the other Embed profile first." >&2
    exit 2
  fi
}
case "$action" in
  start)
    check_embed_conflict
    if [[ "$service" == model ]]; then
      "${compose[@]}" up -d --build --no-recreate "$service"
    else
      "${compose[@]}" up -d --no-build --pull never --no-recreate "$service"
    fi
    ;;
  start-loaded)
    check_embed_conflict
    "${compose[@]}" up -d --no-build --pull never --no-recreate "$service"
    ;;
  restart)
    check_embed_conflict
    if [[ "$service" == model ]]; then
      "${compose[@]}" up -d --build --force-recreate "$service"
    else
      "${compose[@]}" up -d --no-build --pull never --force-recreate "$service"
    fi
    ;;
  restart-loaded)
    check_embed_conflict
    "${compose[@]}" up -d --no-build --pull never --force-recreate "$service"
    ;;
  stop) "${compose[@]}" stop "$service" ;;
  status) "${compose[@]}" ps -a "$service" ;;
  logs) "${compose[@]}" logs --tail 100 -f "$service" ;;
  pull)
    if [[ "$service" == model ]]; then "${compose[@]}" build --pull model
    else
      image=$("${compose[@]}" config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["embed"]["image"])')
      if [[ "$image" == *@sha256:* ]]; then
        echo 'Embed image must be a local tag for docker save/load migration.' >&2
        exit 2
      fi
      upstream=vllm/vllm-openai@sha256:fc56161ee42a011aeee78b65d0a81b6683c7d04402fd40503d14d4d6c98f07cb
      docker pull "$upstream"
      docker image tag "$upstream" "$image"
    fi
    ;;
  build)
    if [[ "$service" == model ]]; then "${compose[@]}" build model
    else echo "Embed uses a pinned upstream image; use pull $group." >&2; exit 2
    fi
    ;;
  config) "${compose[@]}" config --quiet ;;
  check)
    if [[ "$service" == model ]]; then python3 scripts/smoke_test.py
    else python3 scripts/check_embed.py --profile "$group"
    fi
    ;;
  download) python3 scripts/download_model.py --model "$download_model" ;;
  verify) python3 scripts/download_model.py --model "$download_model" --verify-only ;;
  *) echo "Usage: $0 {start|start-loaded|restart|restart-loaded|stop|status|logs|pull|build|config|check|download|verify} [model|embed|embed3|all]" >&2; exit 2 ;;
esac
