#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/bench_common.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/generate_h2h_stats.sh report --input BUNDLE_OR_CHECKPOINT [--report-output MD] [options]
  scripts/generate_h2h_stats.sh run --dataset DATASET --output BUNDLE --checkpoint-dir DIR [options]

Commands:
  report  Render h2h/agreement/cost/stability stats from an existing bundle
          JSON or checkpoint directory.
  run     Generate observations plus h2h games, then render a Markdown report.

Common options:
  --engines LIST          Comma-separated engines, e.g. mcts,minmax
  --h2h-positions N      Positions sampled for h2h (default: 8)
  --h2h-seeds N          H2H seeds (games per pair = positions * seeds * 2)
  --report-output PATH   Markdown report path
  --profile NAME         Cargo profile: release or debug (default: release)
  --dry-run              Print commands without running them
  -h, --help             Show this help

run also accepts the observation engine parameters used by
scripts/generate_observations.sh: --family, --time-limit, --seeds,
--seed-base, --minimax-depth, --minimax-time, --mcts-iterations, --mcts-depth,
--mcts-exploration, --beam-width, --beam-depth, --workers, --checkpoint-every,
and --resume.
USAGE
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

command="$1"
shift
input=""
dataset=""
output=""
checkpoint_dir=""
report_output=""
engines="minimax,mcts,beam,random"
family="native"
time_limit="1.0"
seeds="10"
seed_base="0"
minimax_depth="6"
minimax_time="0.2"
mcts_iterations="1500"
mcts_depth="16"
mcts_exploration="1.414"
beam_width="64"
beam_depth="16"
h2h_positions="8"
h2h_seeds="1"
workers="1"
checkpoint_every="1"
resume="0"
profile="release"
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) input="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --checkpoint-dir) checkpoint_dir="$2"; shift 2 ;;
    --report-output) report_output="$2"; shift 2 ;;
    --engines) engines="$2"; shift 2 ;;
    --family) family="$2"; shift 2 ;;
    --time-limit) time_limit="$2"; shift 2 ;;
    --seeds) seeds="$2"; shift 2 ;;
    --seed-base) seed_base="$2"; shift 2 ;;
    --minimax-depth) minimax_depth="$2"; shift 2 ;;
    --minimax-time) minimax_time="$2"; shift 2 ;;
    --mcts-iterations) mcts_iterations="$2"; shift 2 ;;
    --mcts-depth) mcts_depth="$2"; shift 2 ;;
    --mcts-exploration) mcts_exploration="$2"; shift 2 ;;
    --beam-width) beam_width="$2"; shift 2 ;;
    --beam-depth) beam_depth="$2"; shift 2 ;;
    --h2h-positions) h2h_positions="$2"; shift 2 ;;
    --h2h-seeds) h2h_seeds="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --checkpoint-every) checkpoint_every="$2"; shift 2 ;;
    --resume) resume="1"; shift ;;
    --profile) profile="$2"; shift 2 ;;
    --dry-run) dry_run="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option %q\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$repo_root"
release_flag="$(profile_flag "$profile")"
base=(cargo run)
if [[ -n "$release_flag" ]]; then
  base+=("$release_flag")
fi
base+=(--bin cross_engine_benchmark --)

case "$command" in
  report)
    require_value "--input" "$input"
    report_cmd=("${base[@]}" report --input "$input")
    if [[ -n "$report_output" ]]; then
      report_cmd+=(--output "$report_output")
    fi
    run_or_dry_run "$dry_run" "${report_cmd[@]}"
    ;;
  run)
    require_value "--dataset" "$dataset"
    require_value "--output" "$output"
    require_value "--checkpoint-dir" "$checkpoint_dir"
    engines="$(normalize_engines "$engines")"
    run_cmd=("${base[@]}" run
      --dataset "$dataset"
      --family "$family"
      --time-limit "$time_limit"
      --seeds "$seeds"
      --seed-base "$seed_base"
      --minimax-depth "$minimax_depth"
      --minimax-time "$minimax_time"
      --mcts-iterations "$mcts_iterations"
      --mcts-depth "$mcts_depth"
      --mcts-exploration "$mcts_exploration"
      --beam-width "$beam_width"
      --beam-depth "$beam_depth"
      --engines "$engines"
      --h2h-positions "$h2h_positions"
      --h2h-seeds "$h2h_seeds"
      --checkpoint-dir "$checkpoint_dir"
      --checkpoint-every "$checkpoint_every"
      --workers "$workers"
      --output "$output")
    if [[ "$resume" == "1" ]]; then
      run_cmd+=(--resume)
    fi
    run_or_dry_run "$dry_run" "${run_cmd[@]}"

    report_cmd=("${base[@]}" report --input "$checkpoint_dir")
    if [[ -n "$report_output" ]]; then
      report_cmd+=(--output "$report_output")
    fi
    run_or_dry_run "$dry_run" "${report_cmd[@]}"
    ;;
  -h|--help)
    usage
    ;;
  *)
    printf 'error: unknown command %q\n' "$command" >&2
    usage >&2
    exit 2
    ;;
esac

