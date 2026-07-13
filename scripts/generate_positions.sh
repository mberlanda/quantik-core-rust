#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/bench_common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/generate_positions.sh [options]

Generate a checksummed benchmark position dataset. The Rust dataset generator
deduplicates by canonical key and only emits reachable, valid, non-terminal
positions with at least one legal move.

Options:
  --output PATH          Dataset output path (default: benchmarks/positions-v1.json)
  --opening N           Opening positions, 0-4 plies (default: 8)
  --early-mid N         Early-mid positions, 5-7 plies (default: 8)
  --late-mid N          Late-mid positions, 8-11 plies (default: 12)
  --endgame N           Endgame positions, 12-16 plies (default: 8)
  --seed N              Dataset RNG seed (default: 20260711)
  --solve-budget SECS   Exact-reference solve budget per position (default: 30.0)
  --book PATH           Optional SQLite opening book for reference reuse
  --profile NAME        Cargo profile: release or debug (default: release)
  --dry-run             Print the cargo command without running it
  -h, --help            Show this help
USAGE
}

output="benchmarks/positions-v1.json"
opening="8"
early_mid="8"
late_mid="12"
endgame="8"
seed="20260711"
solve_budget="30.0"
book=""
profile="release"
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    --opening) opening="$2"; shift 2 ;;
    --early-mid) early_mid="$2"; shift 2 ;;
    --late-mid) late_mid="$2"; shift 2 ;;
    --endgame) endgame="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --solve-budget) solve_budget="$2"; shift 2 ;;
    --book) book="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --dry-run) dry_run="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option %q\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$repo_root"
cmd=(cargo run)
release_flag="$(profile_flag "$profile")"
if [[ -n "$release_flag" ]]; then
  cmd+=("$release_flag")
fi
cmd+=(--bin cross_engine_benchmark -- dataset
  --opening "$opening"
  --early-mid "$early_mid"
  --late-mid "$late_mid"
  --endgame "$endgame"
  --seed "$seed"
  --solve-budget "$solve_budget"
  --output "$output")
if [[ -n "$book" ]]; then
  cmd+=(--book "$book")
fi

run_or_dry_run "$dry_run" "${cmd[@]}"

