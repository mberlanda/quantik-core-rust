#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/bench_common.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/plan_runs.sh h2h-games --games N --engines LIST [--positions N] [--seeds N]
  scripts/plan_runs.sh matrix [options]

Commands:
  h2h-games  Calculate h2h_positions and h2h_seeds for a target game count.
             Games per unordered engine pair are positions * seeds * 2.

  matrix     Expand comma-separated parameter lists into dry-run commands for
             scripts/generate_h2h_stats.sh run. This is intentionally plain
             text so a later TUI can consume or replace it.

h2h-games options:
  --games N       Target number of h2h games
  --engines LIST  Comma-separated engines, e.g. mcts,minmax
  --positions N   H2H positions to sample. If omitted, defaults to 50.
  --seeds N       H2H seeds. If omitted, calculated with ceiling division.

matrix options:
  --dataset PATH            Dataset path (default: benchmarks/positions-v1.json)
  --output-dir DIR          Bundle/checkpoint/report directory (default: benchmarks/results/matrix)
  --engines LISTS           Semicolon-separated engine lists (default: mcts,minimax)
  --games N                 Target h2h games per run (default: 1000)
  --positions N             H2H positions per run (default: 50)
  --mcts-iterations LIST    Comma-separated values (default: 1500)
  --minimax-depth LIST      Comma-separated values (default: 6)
  --beam-width LIST         Comma-separated values (default: 64)
  --seeds LIST              Observation seed counts (default: 10)
  -h, --help                Show this help
USAGE
}

join_slug() {
  printf '%s' "$1" | tr ',;=' '---' | tr -cd '[:alnum:]_.-'
}

split_csv_values() {
  local raw="$1"
  IFS=',' read -r -a split_values <<< "$raw"
}

command="${1:-}"
if [[ -z "$command" ]]; then
  usage >&2
  exit 2
fi
shift || true

case "$command" in
  h2h-games)
    games=""
    engines=""
    positions=""
    seeds=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --games) games="$2"; shift 2 ;;
        --engines) engines="$2"; shift 2 ;;
        --positions) positions="$2"; shift 2 ;;
        --seeds) seeds="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown option %q\n' "$1" >&2; usage >&2; exit 2 ;;
      esac
    done
    require_value "--games" "$games"
    require_value "--engines" "$engines"
    engines="$(normalize_engines "$engines")"
    engine_count="$(count_csv "$engines")"
    if [[ "$engine_count" -lt 2 ]]; then
      printf 'error: h2h planning requires at least two engines\n' >&2
      exit 2
    fi
    pair_count=$((engine_count * (engine_count - 1) / 2))
    if [[ -z "$positions" ]]; then
      positions="50"
    fi
    if [[ -z "$seeds" ]]; then
      seeds="$(ceil_div "$games" "$((positions * pair_count * 2))")"
    fi
    planned=$((positions * seeds * pair_count * 2))
    printf 'engines=%s\n' "$engines"
    printf 'engine_pairs=%s\n' "$pair_count"
    printf 'h2h_positions=%s\n' "$positions"
    printf 'h2h_seeds=%s\n' "$seeds"
    printf 'planned_games=%s\n' "$planned"
    printf 'cargo_args=--engines %s --h2h-positions %s --h2h-seeds %s\n' \
      "$engines" "$positions" "$seeds"
    ;;
  matrix)
    dataset="benchmarks/positions-v1.json"
    output_dir="benchmarks/results/matrix"
    engine_sets="mcts,minimax"
    games="1000"
    positions="50"
    mcts_iterations_values="1500"
    minimax_depth_values="6"
    beam_width_values="64"
    observation_seed_values="10"
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --dataset) dataset="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --engines) engine_sets="$2"; shift 2 ;;
        --games) games="$2"; shift 2 ;;
        --positions) positions="$2"; shift 2 ;;
        --mcts-iterations) mcts_iterations_values="$2"; shift 2 ;;
        --minimax-depth) minimax_depth_values="$2"; shift 2 ;;
        --beam-width) beam_width_values="$2"; shift 2 ;;
        --seeds) observation_seed_values="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown option %q\n' "$1" >&2; usage >&2; exit 2 ;;
      esac
    done

    IFS=';' read -r -a engine_set_values <<< "$engine_sets"
    split_csv_values "$mcts_iterations_values"; mcts_values=("${split_values[@]}")
    split_csv_values "$minimax_depth_values"; minimax_values=("${split_values[@]}")
    split_csv_values "$beam_width_values"; beam_values=("${split_values[@]}")
    split_csv_values "$observation_seed_values"; obs_seed_values=("${split_values[@]}")

    run_id=0
    for raw_engines in "${engine_set_values[@]}"; do
      engines="$(normalize_engines "$raw_engines")"
      engine_count="$(count_csv "$engines")"
      pair_count=$((engine_count * (engine_count - 1) / 2))
      h2h_seeds="$(ceil_div "$games" "$((positions * pair_count * 2))")"
      for mcts_iterations in "${mcts_values[@]}"; do
        for minimax_depth in "${minimax_values[@]}"; do
          for beam_width in "${beam_values[@]}"; do
            for obs_seeds in "${obs_seed_values[@]}"; do
              run_id=$((run_id + 1))
              slug="$(join_slug "${engines}_mcts${mcts_iterations}_mm${minimax_depth}_beam${beam_width}_obs${obs_seeds}")"
              printf 'run=%04d engines=%s planned_games=%s\n' \
                "$run_id" "$engines" "$((positions * h2h_seeds * pair_count * 2))"
              print_cmd scripts/generate_h2h_stats.sh run \
                --dataset "$dataset" \
                --output "${output_dir}/${slug}.json" \
                --checkpoint-dir "${output_dir}/${slug}-ckpt" \
                --report-output "${output_dir}/${slug}.md" \
                --engines "$engines" \
                --h2h-positions "$positions" \
                --h2h-seeds "$h2h_seeds" \
                --mcts-iterations "$mcts_iterations" \
                --minimax-depth "$minimax_depth" \
                --beam-width "$beam_width" \
                --seeds "$obs_seeds"
            done
          done
        done
      done
    done
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

