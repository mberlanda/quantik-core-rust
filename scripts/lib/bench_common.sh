#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

profile_flag() {
  case "${1:-release}" in
    release) printf '%s\n' "--release" ;;
    debug) printf '%s\n' "" ;;
    *)
      printf 'error: unknown profile %q; use release or debug\n' "$1" >&2
      exit 2
      ;;
  esac
}

normalize_engine() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    minimax|minmax) printf '%s\n' "minimax" ;;
    mcts) printf '%s\n' "mcts" ;;
    beam|beam_search|beam-search) printf '%s\n' "beam" ;;
    random|baseline) printf '%s\n' "random" ;;
    *)
      printf 'error: unknown engine %q; supported: minimax, mcts, beam, random\n' "$1" >&2
      exit 2
      ;;
  esac
}

normalize_engines() {
  local raw="${1:-minimax,mcts,beam,random}"
  local normalized=""
  local engine name
  IFS=',' read -r -a engines <<< "$raw"
  for engine in "${engines[@]}"; do
    name="$(normalize_engine "$engine")"
    case ",${normalized}," in
      *",${name},"*)
        printf 'error: duplicate engine %q\n' "$name" >&2
        exit 2
        ;;
    esac
    if [[ -z "$normalized" ]]; then
      normalized="$name"
    else
      normalized="${normalized},${name}"
    fi
  done
  printf '%s\n' "$normalized"
}

count_csv() {
  local csv="$1"
  if [[ -z "$csv" ]]; then
    printf '%s\n' "0"
    return
  fi
  local count=1
  local rest="$csv"
  while [[ "$rest" == *,* ]]; do
    count=$((count + 1))
    rest="${rest#*,}"
  done
  printf '%s\n' "$count"
}

ceil_div() {
  local numerator="$1"
  local denominator="$2"
  if [[ "$denominator" -le 0 ]]; then
    printf 'error: denominator must be positive\n' >&2
    exit 2
  fi
  printf '%s\n' $(((numerator + denominator - 1) / denominator))
}

print_cmd() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf '\n'
}

run_or_dry_run() {
  local dry_run="$1"
  shift
  if [[ "$dry_run" == "1" ]]; then
    print_cmd "$@"
  else
    "$@"
  fi
}

cross_engine_cmd() {
  local profile="${1:-release}"
  local flag
  flag="$(profile_flag "$profile")"
  if [[ -n "$flag" ]]; then
    print_cmd cargo run "$flag" --bin cross_engine_benchmark --
  else
    print_cmd cargo run --bin cross_engine_benchmark --
  fi
}

require_value() {
  local name="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    printf 'error: %s is required\n' "$name" >&2
    exit 2
  fi
}
