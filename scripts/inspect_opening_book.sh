#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/bench_common.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/inspect_opening_book.sh stats --db DB [options]
  scripts/inspect_opening_book.sh frontier --db DB [options]
  scripts/inspect_opening_book.sh storage --db DB [options]
  scripts/inspect_opening_book.sh resume-command --db DB --depth N [options]

Commands:
  stats           Print per-depth positions, terminal rows, edges, symmetry,
                  searched-depth, and rows still needing search.
  frontier        Print sample non-terminal rows that need more search for a
                  target depth.
  storage         Print SQLite page/object storage details.
  resume-command  Print the opening-book search command that resumes DB to N.

Options:
  --db PATH              bench_bfs SQLite database
  --depth N             Target depth for stats/frontier/resume-command
  --limit N             Frontier sample size
  --profile NAME        Cargo profile: release or debug (default: release)
  --dry-run             Print the cargo command without running it
  -h, --help            Show this help
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
db=""
depth=""
limit=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db) db="$2"; shift 2 ;;
    --depth|--target-depth) depth="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --dry-run) dry_run="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown option %q\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$command" in
  stats|frontier|storage|resume-command) ;;
  -h|--help) usage; exit 0 ;;
  *) printf 'error: unknown command %q\n' "$command" >&2; usage >&2; exit 2 ;;
esac

require_value "--db" "$db"

if [[ "$command" == "resume-command" ]]; then
  require_value "--depth" "$depth"
  printf 'scripts/generate_opening_book.sh search --depth %s --db %s --resume\n' "$depth" "$db"
  exit 0
fi

cd "$repo_root"
cmd=(cargo run)
release_flag="$(profile_flag "$profile")"
if [[ -n "$release_flag" ]]; then
  cmd+=("$release_flag")
fi

cmd+=(--bin bench_bfs_inspect -- --db "$db" "$command")
if [[ -n "$depth" ]]; then
  cmd+=(--target-depth "$depth")
fi
if [[ -n "$limit" ]]; then
  cmd+=(--limit "$limit")
fi

run_or_dry_run "$dry_run" "${cmd[@]}"
