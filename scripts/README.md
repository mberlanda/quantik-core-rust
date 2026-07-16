# Quantik Rust Generation Scripts

These scripts are a small, stable tool layer over the Rust
`cross_engine_benchmark`, `bench_bfs`, and opening-book binaries.

Run every command from the repository root.

## Positions

Generate valid, reachable, non-terminal positions:

```sh
scripts/generate_positions.sh \
  --opening 250 --early-mid 250 --late-mid 250 --endgame 250 \
  --output benchmarks/positions-1000.json
```

The underlying Rust dataset generator deduplicates positions by canonical key
and rejects terminal/dead-end positions.

Pass `--book PATH` to reuse (and write back) exact solved references from an
opening-book SQLite database. Both book flavors work: a benchmark book built
with `generate_opening_book.sh export`, and a searched book built with
`generate_opening_book.sh search` (`bench_bfs`). A searched book is upgraded
in place on open — the missing benchmark-book columns are added with default
values — so its searched rows are never served as solved references, while
newly solved positions are written back into the same file:

```sh
scripts/generate_positions.sh \
  --opening 250 --early-mid 250 --late-mid 250 --endgame 250 \
  --book benchmarks/results/depth6-book.sqlite \
  --output benchmarks/positions-1000.json
```

## Opening Book

Export exact solved references from a positions dataset:

```sh
scripts/generate_opening_book.sh export \
  --input benchmarks/positions-1000.json \
  --db benchmarks/results/opening-book.sqlite
```

Or build/search an opening book directly with the IDDFS builder:

```sh
scripts/generate_opening_book.sh search \
  --depth 6 \
  --db benchmarks/results/depth6-book.sqlite
```

Inspect an existing `bench_bfs` SQLite book:

```sh
scripts/inspect_opening_book.sh stats \
  --db benchmarks/results/depth6-book.sqlite \
  --depth 7

scripts/inspect_opening_book.sh frontier \
  --db benchmarks/results/depth6-book.sqlite \
  --depth 7 \
  --limit 20

scripts/inspect_opening_book.sh storage \
  --db benchmarks/results/depth6-book.sqlite
```

To continue expanding non-terminal horizon rows, resume to a larger target
depth:

```sh
scripts/generate_opening_book.sh search \
  --depth 7 \
  --db benchmarks/results/depth6-book.sqlite \
  --resume
```

## Observations

Generate observations for selected engines with checkpointing:

```sh
scripts/generate_observations.sh \
  --dataset benchmarks/positions-1000.json \
  --output benchmarks/results/mcts-minimax-observations.json \
  --checkpoint-dir benchmarks/results/mcts-minimax-observations-ckpt \
  --engines mcts,minimax \
  --mcts-iterations 5000 \
  --minimax-depth 8 \
  --seeds 30 \
  --workers 4
```

By default this script passes `--skip-h2h`, so it records observation rows
without playing head-to-head games. Add `--include-h2h` when you want one run
to produce both observations and games.

Export contract-owned rows from the resulting bundle or checkpoint:

```sh
scripts/export_contract_rows.sh \
  --input benchmarks/results/mcts-minimax-observations-ckpt \
  --dataset benchmarks/positions-1000.json \
  --observations-output benchmarks/results/observations-v1.jsonl
```

For H2H runs, also export completed games:

```sh
scripts/export_contract_rows.sh \
  --input benchmarks/results/mcts-vs-minimax-ckpt \
  --dataset benchmarks/positions-1000.json \
  --observations-output benchmarks/results/observations-v1.jsonl \
  --games-output benchmarks/results/game-results-v1.jsonl
```

## H2H Stats

Plan the parameters for 1000 games between MCTS and minimax:

```sh
scripts/plan_runs.sh h2h-games \
  --games 1000 \
  --engines mcts,minmax \
  --positions 50
```

This prints:

```text
engines=mcts,minimax
engine_pairs=1
h2h_positions=50
h2h_seeds=10
planned_games=1000
cargo_args=--engines mcts,minimax --h2h-positions 50 --h2h-seeds 10
```

Generate observations, h2h games, and a Markdown report:

```sh
scripts/generate_h2h_stats.sh run \
  --dataset benchmarks/positions-1000.json \
  --output benchmarks/results/mcts-vs-minimax.json \
  --checkpoint-dir benchmarks/results/mcts-vs-minimax-ckpt \
  --report-output benchmarks/results/mcts-vs-minimax.md \
  --engines mcts,minmax \
  --h2h-positions 50 \
  --h2h-seeds 10 \
  --mcts-iterations 5000 \
  --minimax-depth 8 \
  --workers 4
```

Render stats from an existing bundle or checkpoint directory:

```sh
scripts/generate_h2h_stats.sh report \
  --input benchmarks/results/mcts-vs-minimax-ckpt \
  --report-output benchmarks/results/mcts-vs-minimax.md
```

## Search Telemetry Export

Export draft `search-summary.v1-draft` JSONL rows (event counters, root-move
statistics, principal variation) for MCTS, beam, and minimax over a small
fixed position set — see `docs/search-telemetry.md` for the event semantics
and value-mapping details:

```sh
cargo run -p quantik-core --example search_summary_export -- \
  --out benchmarks/results/search-summaries.jsonl
```

## Parameter Matrices

Expand combinations into runnable commands:

```sh
scripts/plan_runs.sh matrix \
  --engines 'mcts,minimax;mcts,beam;minimax,beam' \
  --games 1000 \
  --positions 50 \
  --mcts-iterations 1500,5000 \
  --minimax-depth 6,8 \
  --beam-width 64,128
```

Use `--dry-run` on the generation scripts when wiring this into a TUI or job
runner.
