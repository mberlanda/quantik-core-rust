#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/bench_common.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/generate_opening_book.sh export --input DATASET --db DB [options]
  scripts/generate_opening_book.sh search --depth N --db DB [options]

Commands:
  export   Export solved references from a generated positions dataset into
           an opening-book SQLite database.
  search   Build/search an opening book with the bench_bfs IDDFS builder.

Options:
  --profile NAME        Cargo profile: release or debug (default: release)
  --dry-run             Print the cargo command without running it
  -h, --help            Show this help

search options:
  --resume              Resume an existing bench_bfs database
  --max-positions N     Stop after N total positions
  --exhaustive-depth N  Exhaustive phase depth
  --batch-size N        SQLite transaction batch size
  --quiet               Only print final bench_bfs summary
USAGE
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

command="$1"
shift
profile="release"
dry_run="0"
input=""
db=""
depth=""
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) input="$2"; shift 2 ;;
    --db) db="$2"; shift 2 ;;
    --depth) depth="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --resume|--quiet) extra+=("$1"); shift ;;
    --max-positions|--exhaustive-depth|--batch-size) extra+=("$1" "$2"); shift 2 ;;
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

case "$command" in
  export)
    require_value "--input" "$input"
    require_value "--db" "$db"
    cmd+=(--bin cross_engine_benchmark -- export-book --input "$input" --db "$db")
    ;;
  search)
    require_value "--depth" "$depth"
    require_value "--db" "$db"
    cmd+=(--bin bench_bfs -- "$depth" --db "$db")
    if [[ ${#extra[@]} -gt 0 ]]; then
      cmd+=("${extra[@]}")
    fi
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    printf 'error: unknown command %q\n' "$command" >&2
    usage >&2
    exit 2
    ;;
esac

run_or_dry_run "$dry_run" "${cmd[@]}"

