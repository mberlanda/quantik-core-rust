#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/bench_common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/export_contract_rows.sh --input BUNDLE_OR_CHECKPOINT --dataset DATASET [options]

Project benchmark bundles/checkpoints into contracts-owned JSONL rows:
observation.v1 for engine observations and game-result.v1 for h2h games.

Options:
  --input PATH                 Bundle JSON or checkpoint directory
  --dataset PATH               Positions dataset used by the benchmark
  --observations-output PATH   observation.v1 JSONL output
  --games-output PATH          game-result.v1 JSONL output
  --profile NAME               Cargo profile: release or debug (default: release)
  --dry-run                    Print cargo commands without running them
  -h, --help                   Show this help
USAGE
}

input=""
dataset=""
observations_output=""
games_output=""
profile="release"
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) input="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --observations-output) observations_output="$2"; shift 2 ;;
    --games-output) games_output="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --dry-run) dry_run="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option %q\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

require_value "--input" "$input"
require_value "--dataset" "$dataset"
if [[ -z "$observations_output" && -z "$games_output" ]]; then
  printf 'error: at least one of --observations-output or --games-output is required\n' >&2
  exit 2
fi

cd "$repo_root"
cmd_prefix=(cargo run)
release_flag="$(profile_flag "$profile")"
if [[ -n "$release_flag" ]]; then
  cmd_prefix+=("$release_flag")
fi
cmd_prefix+=(--bin cross_engine_benchmark --)

if [[ -n "$observations_output" ]]; then
  run_or_dry_run "$dry_run" "${cmd_prefix[@]}" export-observations \
    --input "$input" \
    --dataset "$dataset" \
    --output "$observations_output"
fi

if [[ -n "$games_output" ]]; then
  run_or_dry_run "$dry_run" "${cmd_prefix[@]}" export-games \
    --input "$input" \
    --dataset "$dataset" \
    --output "$games_output"
fi
