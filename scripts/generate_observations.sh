#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/bench_common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/generate_observations.sh --dataset DATASET --output BUNDLE --checkpoint-dir DIR [options]

Run engine observations for a generated dataset. By default this skips h2h
games so checkpoint rows focus on observations. Pass --include-h2h when you
want the same run to also generate head-to-head games.

Options:
  --dataset PATH          Input positions dataset
  --output PATH           Result bundle JSON
  --checkpoint-dir DIR    Directory checkpoint for observations/h2h rows
  --engines LIST          Comma-separated engines, e.g. mcts,minmax,beam
                          (default: minimax,mcts,beam,random)
  --family NAME           fixed or native (default: native)
  --time-limit SECS       Fixed-family time per move (default: 1.0)
  --seeds N               Stochastic observation seeds (default: 10)
  --seed-base N           First seed value (default: 0)
  --minimax-depth N       Native minimax depth (default: 6)
  --minimax-time SECS     Native minimax time cap (default: 0.2)
  --mcts-iterations N     Native MCTS iterations (default: 1500)
  --mcts-depth N          Native MCTS rollout/search depth (default: 16)
  --mcts-exploration X    Native MCTS exploration weight (default: 1.414)
  --beam-width N          Native/fixed beam width (default: 64)
  --beam-depth N          Native beam depth (default: 16)
  --workers N             Parallel workers (default: 1)
  --checkpoint-every N    Manifest/progress interval (default: 1)
  --resume                Resume an existing checkpoint
  --include-h2h           Also generate h2h games in this run
  --profile NAME          Cargo profile: release or debug (default: release)
  --dry-run               Print the cargo command without running it
  -h, --help              Show this help
USAGE
}

dataset=""
output=""
checkpoint_dir=""
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
workers="1"
checkpoint_every="1"
resume="0"
include_h2h="0"
profile="release"
dry_run="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) dataset="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --checkpoint-dir) checkpoint_dir="$2"; shift 2 ;;
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
    --workers) workers="$2"; shift 2 ;;
    --checkpoint-every) checkpoint_every="$2"; shift 2 ;;
    --resume) resume="1"; shift ;;
    --include-h2h) include_h2h="1"; shift ;;
    --profile) profile="$2"; shift 2 ;;
    --dry-run) dry_run="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option %q\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

require_value "--dataset" "$dataset"
require_value "--output" "$output"
require_value "--checkpoint-dir" "$checkpoint_dir"
engines="$(normalize_engines "$engines")"

cd "$repo_root"
cmd=(cargo run)
release_flag="$(profile_flag "$profile")"
if [[ -n "$release_flag" ]]; then
  cmd+=("$release_flag")
fi
cmd+=(--bin cross_engine_benchmark -- run
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
  --checkpoint-dir "$checkpoint_dir"
  --checkpoint-every "$checkpoint_every"
  --workers "$workers"
  --output "$output")
if [[ "$resume" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "$include_h2h" != "1" ]]; then
  cmd+=(--skip-h2h)
fi

run_or_dry_run "$dry_run" "${cmd[@]}"

