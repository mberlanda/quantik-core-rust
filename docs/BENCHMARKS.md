# Cross-Engine Benchmark Methodology

The `cross_engine_benchmark` binary compares `MinimaxEngine`, `MCTSEngine`,
`BeamSearchEngine`, and a uniform-random baseline under reproducible,
methodologically consistent conditions. It reproduces the Python
`examples/cross_engine_benchmark.py` harness on the same shared dataset.

The harness separates four questions that a single timing or win-rate
number would conflate:

1. Move quality: does the engine select an objectively optimal move?
2. Playing strength: how do engines perform against one another?
3. Computational cost: measured time, nodes, iterations, and memory.
4. Stability: do stochastic engines behave consistently across seeds?

## Shared Dataset

All engines are evaluated on exactly the same positions, generated once by
the `dataset` subcommand and saved as `benchmarks/positions-v1.json`. The
artifact records the generation seed, generator id, schema version, and a
sha256 checksum that `run` verifies on load.

Positions are valid, reachable, non-terminal, globally deduplicated by
`State::canonical_key()`, and record the side to move. Phase buckets are by
pieces placed, which is the same as plies from the empty board:

- opening: 0-4
- early_mid: 5-7
- late_mid: 8-11
- endgame: 12-16

The committed `benchmarks/positions-v1.json` is the artifact shared with the
Python implementation: byte-compatible JSON schema, same checksum algorithm,
and the same 18-byte `pack`/`canonical_key` binary state format (`VERSION=1`,
`FLAG_CANON=2`, little-endian `<BB8H>`). Regenerating the dataset from
scratch with `dataset` will *not* reproduce the same positions in each
language, because the two languages' RNG streams differ even with an
identical seed — the committed artifact, not a freshly generated one, is the
cross-language contract.

## Exact References

Non-opening positions may carry an exact reference: game value for the side
to move plus the complete set of optimal moves. References are produced by
full-depth minimax and stored only when every child was solved with no
cutoff. Quantik never exceeds 16 plies, so a completed iterative-deepening
depth at least equal to the child's remaining plies proves exactness.

Positions that exceed the per-position solve budget, and the whole opening
bucket, have no exact reference and never contribute to exact move-agreement
figures. An engine scores a hit when its selected move is in the complete
optimal set, not merely equal to one arbitrary principal-variation move.

## Benchmark Families

- `fixed`: every engine gets the same wall-clock budget per move. This is
  the fair practical-latency comparison. Beam search checks its deadline
  between depth levels, so a wide level can overshoot; compare measured
  wall times, not configured caps.
- `native`: each engine runs with explicit native settings such as minimax
  depth/time, MCTS iterations/depth/exploration, and beam width/depth. This
  explains scaling behavior but is not a fair head-to-head ranking.

Every generated bundle and Markdown report records which family was run.

## Stochastic Engines

MCTS, beam, and random are seed-sensitive; minimax is deterministic. The
`run` subcommand evaluates stochastic engines across `--seeds N` seeds
using the same ordered seed list for every stochastic adapter. Stability is
computed from the same raw agreement rows, so engines are not rerun just to
produce another table.

The stability table reports:

- move consistency: the average fraction of seeds choosing the modal move
  per position
- agreement mean/std: per-seed exact-reference agreement, then summarized
  across seeds

Use at least 10 seeds during development and at least 30 for publishable
results.

## Head-To-Head

Every sampled position is played twice per seed: each engine once as the
side already to move. Wins are credited to the actual engine/color mapping,
because sampled positions can have either player to move. Quantik has no
draws, but the bundle still carries `draws: 0` so reports have a stable
shape.

## Correctness Preflight

`run` refuses to benchmark until preflight invariants pass:

- dataset positions are non-terminal
- every adapter returns a legal move for the correct side
- adapters do not mutate their input bitboard
- identical settings and seed reproduce the same move
- minimax's chosen move matches the head of its principal variation

## Checkpoint And Resume

`run` accepts `--checkpoint <path>` to stream every completed observation
row and head-to-head game to a JSON Lines file as it happens, and
`--resume` to continue an interrupted run from that file instead of
restarting it from scratch. This is a Rust-only addition — the Python
harness has no equivalent.

```bash
cargo run --release --bin cross_engine_benchmark -- run \
  --dataset benchmarks/positions-v1.json --time-limit 1.0 --seeds 30 \
  --checkpoint benchmarks/results/run.ckpt \
  --output benchmarks/results/run.json
# interrupted (Ctrl-C, crash, machine reboot, ...) partway through
cargo run --release --bin cross_engine_benchmark -- run \
  --dataset benchmarks/positions-v1.json --time-limit 1.0 --seeds 30 \
  --checkpoint benchmarks/results/run.ckpt --resume \
  --output benchmarks/results/run.json
```

**Crash-safety guarantee:** the checkpoint file is a header line followed
by one line per completed row/game (`{"kind":"observation","row":{...}}` or
`{"kind":"h2h","record":{...}}`), each flushed to disk individually. A
crash can lose at most the single line that was in flight — every line
written before it is intact, and a truncated trailing line is detected and
dropped on the next load rather than corrupting the read. Resuming
re-truncates any such dangling line before appending further, so the file
never accumulates garbage in its middle.

**Header validation:** the header records `dataset_checksum` (the loaded
dataset's checksum) and `config_fingerprint` (a sha256 of the run's
canonical config JSON, excluding the `--output`/`--checkpoint` paths
themselves — so the same engine settings resumed to a different output
path still match). `--resume` refuses to proceed if either mismatches the
current run's dataset or settings, so runs are never silently mixed;
without `--resume`, `run` refuses to overwrite an existing checkpoint file
at all. On a successful resumed run, the result bundle's top-level
`"resumed"` field is `true` (`false` for every non-resumed run), and the
`observations`/`head_to_head.records` arrays contain the union of the
checkpoint's rows/games and any newly completed ones — order is not
preserved, but no row is duplicated or lost.

## Reproducing A Run

Generate or update the committed dataset artifact:

```bash
cargo run --release --bin cross_engine_benchmark -- dataset \
  --opening 8 --early-mid 8 --late-mid 12 --endgame 8 \
  --seed 20260711 --solve-budget 30.0 \
  --output benchmarks/positions-v1.json
```

Run the recommended fixed-resource benchmark:

```bash
cargo run --release --bin cross_engine_benchmark -- run \
  --dataset benchmarks/positions-v1.json \
  --time-limit 1.0 --seeds 30 \
  --output benchmarks/results/$(git rev-parse --short HEAD).json
cargo run --release --bin cross_engine_benchmark -- report \
  --input benchmarks/results/$(git rev-parse --short HEAD).json
```

`benchmarks/results/` is gitignored. Attach reports to PRs or issues
instead of committing them.

## Environment Record

Each bundle's `environment` block fingerprints the host and toolchain that
produced it: `quantik_core_version` (from `CARGO_PKG_VERSION`), `git_sha`
(from `git rev-parse HEAD`), `rust_version` (the actual `rustc` version in
use, not the crate's declared MSRV), `platform`, `processor`, and
`cpu_count`. This takes the place of the Python bundle's `python_version`
field — the rest of the schema (`schema_version`, `started_at`, `config`,
`dataset`, `observations`, `head_to_head`, `aggregates`) is identical.

## Interpretation Guardrails

Minimax buys adversarial certainty when the remaining tree is small enough.
MCTS buys empirical confidence through repeated sampling. Beam search buys
bounded, selectively deep exploration. Claims that one engine is universally
superior require evidence across multiple phases, equivalent budgets,
repeated seeds, and statistically meaningful samples.
