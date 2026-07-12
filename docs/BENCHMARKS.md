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

`run` accepts `--checkpoint-dir <dir>` to stream every completed
observation row and head-to-head game into a **directory** as they
complete, and `--resume` to continue an interrupted run from that
directory instead of restarting it from scratch. The layout is
byte-compatible with the Python harness's own `benchmarks/checkpoint.py`
directory format (added there 2026-07-12), so a checkpoint directory
written by this crate loads and reports correctly in Python and vice
versa:

```text
<checkpoint-dir>/
  manifest.json       pretty (indent 2), sorted-key JSON + trailing "\n",
                       written atomically (tmp file + rename)
  observations.jsonl  one compact, sorted-key JSON row per completed
                       agreement observation, appended + flushed per line
  h2h.jsonl            one compact, sorted-key JSON row per completed
                       head-to-head game, appended + flushed per line
```

```bash
cargo run --release --bin cross_engine_benchmark -- run \
  --dataset benchmarks/positions-v1.json --time-limit 1.0 --seeds 30 \
  --checkpoint-dir benchmarks/results/run-ckpt \
  --output benchmarks/results/run.json
# interrupted (Ctrl-C, crash, machine reboot, ...) partway through
cargo run --release --bin cross_engine_benchmark -- run \
  --dataset benchmarks/positions-v1.json --time-limit 1.0 --seeds 30 \
  --checkpoint-dir benchmarks/results/run-ckpt --resume \
  --output benchmarks/results/run.json
```

This replaces the single-file `.ckpt` format from the previous release
(`--checkpoint <path>`): the directory format is the one that
interoperates with Python, so the old format was removed rather than kept
alongside it. There is no automatic migration; a `.ckpt` file from the
old format cannot be resumed under the new layout.

**Partial-state reporting:** `report --input <dir>` accepts a checkpoint
directory directly — no separate "finish the run" step is needed to see
where things stand. It rehydrates the manifest and both JSONL files into
the standard bundle shape (recomputing aggregates from whatever rows/games
exist so far) and adds a `"checkpoint": {"status": ..., "counts": {...}}`
block; the rendered Markdown gets extra lines after `- started:`:

```text
- checkpoint status: running
- checkpoint counts: observations 1840, h2h_records 96
```

**Crash-safety guarantee:** `manifest.json` is the integrity anchor — it
is always written atomically (a `.tmp` file, then renamed over the real
path), so it is never observed half-written. Each JSONL row is appended
and flushed to the OS individually, so a crash mid-write can only ever
leave a trailing partial line in `observations.jsonl`/`h2h.jsonl`, never a
torn manifest. Unlike the old single-file format, a corrupt/partial JSONL
line is a hard load error (`<path>:<line>: invalid checkpoint JSON`)
rather than being silently tolerated — this matches the Python format's
behavior, which treats the atomic manifest as the sole source of "how far
did this get," not per-line tail tolerance.

**Manifest validation on resume:** the manifest records the dataset
checksum and the full run config (including `engine_seeds`). `--resume`
refuses to proceed (`RESUME FAILED - ...`) if the checkpoint's manifest is
missing, or if the dataset checksum or the *normalized* config differ from
this run's — the normalization drops `checkpoint_dir`, `output`,
`resume`, and `workers` (fields that legitimately differ between an
interrupted run and its resume) — and the error names every mismatched
config key with its expected/found values. Without `--resume`,
`--checkpoint-dir` always starts a fresh directory: it truncates
`observations.jsonl`/`h2h.jsonl` and overwrites `manifest.json`
unconditionally (matching the Python harness — there is no "refuse to
clobber an existing directory" guard).

**Resume validation is intra-language only.** The Rust and Python CLIs
don't parse the same flag set (e.g. Python's `--track-memory`/
`--skip-agreement` have no Rust equivalent), so their run-config
dictionaries differ in shape even for equivalent runs — a checkpoint
directory started by one language's `run` is not expected to `--resume`
cleanly under the other. Loading and reporting
(`bundle_from_checkpoint`/`report --input <dir>`) are unaffected by this
and work identically regardless of which language wrote the directory,
since they only read the manifest's stored data back, never compare it
against a new run's config.

**Preflight and progress output:** `run` prints its progress to stdout,
flushed after every line, mirroring the Python CLI's wording:

```text
preflight: checking 4 adapters across 3 sample positions
preflight: passed
agreement: 0/1080 observations complete; workers=4; checkpoint benchmarks/results/run-ckpt/observations.jsonl
agreement: 20/1080 observations checkpointed
...
h2h: 0/1440 games complete; workers=4; checkpoint benchmarks/results/run-ckpt/h2h.jsonl
h2h: 20/1440 games checkpointed
...
bundle: 1080 observations, 1440 games -> benchmarks/results/run.json
```

A preflight failure updates the manifest's `status` to
`"preflight_failed"` (keeping whatever counts existed) before the process
exits 1, so `report --input <dir>` can still show what happened.

## Parallel Workers

`run --workers N` (default 1) runs agreement observations and
head-to-head games across an `N`-thread rayon pool instead of serially.
The task list (every `(adapter, position, seed)` unit for agreement, every
`(mover, responder, position, seed)` unit for head-to-head, in the same
order Python's `_agreement_tasks`/`_h2h_tasks` iterate) is built up front,
skip-filtered for anything already in the checkpoint, then either run
serially (`workers == 1`) or dispatched to the pool via
`par_iter().collect()` — which preserves input order in its result
`Vec` — and streamed to the checkpoint/bundle in that same order
afterward. Because of this, **`--workers N` produces results in the exact
same order, and (for adapters whose behavior doesn't depend on real
wall-clock CPU time) the exact same content, as `--workers 1`** — it is a
throughput knob, not a source of nondeterminism in run structure. An
adapter error inside a worker fails the whole run, exactly as it would
serially; the checkpoint keeps whatever it had already written.

**Caveat for wall-clock time-limited adapters:** `--time-limit`
(fixed family), `--minimax-time`, and any other engine setting that bounds
search by wall-clock deadline rather than a fixed iteration/depth count is
inherently sensitive to real CPU contention. Running several such searches
concurrently under `--workers N > 1` means each one gets a smaller share
of the CPU within its nominal time budget than it would running alone, so
it may legitimately complete less search (a shallower `depth_reached`,
fewer `nodes`) than the same task would under `--workers 1` — the same
variance a busy machine introduces between two serial runs. This is a
property of wall-clock deadlines under real parallelism, not a bug in
task ordering; the order and adapter/position/seed identity of every row
are unaffected.

## Opening Book Persistence

Exactly-solved references can be persisted into the existing SQLite
opening book and reused across runs, so repeated dataset builds stop
recomputing identical solve trees. This is a Rust-only addition — the
Python harness recomputes every reference from scratch.

**What is stored:** one row per solved position in the book's `positions`
table, keyed by the 18-byte canonical key (`State::canonical_key()`),
with `solved = 1`, `game_value` = the exact game value for the side to
move (+1/-1, stored as-is), `evaluation` = the game value as a float,
`depth` = pieces placed, and the complete optimal move set as
`(shape, position)` pairs in the `best_moves` table — recorded in the
canonical orientation. Pre-existing book databases are upgraded in place:
opening one adds the `solved`/`game_value` columns idempotently, and old
rows read back as `solved = false`.

**Representative-only caveat (applies to both reads and writes):** the
stored optimal moves are `(shape, position)` pairs in one specific board
orientation, but the row is keyed by the canonical key, which is shared
by up to eight symmetric orientations. The book does not (yet) record
which symmetry transform maps the stored orientation to an arbitrary
query, so moves cannot be translated across orientations. Both
directions are therefore restricted to boards that are their own
canonical representative:

- *Writes* skip any solved position that is not its own canonical
  representative — storing one would let a later lookup on the
  representative board pass the read check and be served moves that are
  wrong (possibly illegal) for it. `export-book` counts and inserts only
  the representative subset (a solved reference that is skipped is simply
  re-solvable later).
- *Reads* only return a hit when the queried board is its own canonical
  representative — any other orientation falls through to a fresh
  minimax solve (which then writes back if, and only if, the board is a
  representative).

Move translation via the stored symmetry transform is the documented
follow-up that would lift the restriction on both sides; until then,
book hits accelerate the canonical-representative subset of queries and
never risk serving moves that are illegal in the queried orientation.

**Cross-language portability:** the canonical key is byte-identical to
the Python implementation's (`VERSION=1`, `FLAG_CANON=2`, little-endian
`<BB8H>`), and the schema is the same family as `opening_book.py`'s, so
the SQLite file itself is portable — a book populated by
`quantik-core-rust` is readable from Python and vice versa.

Two CLI entry points:

```bash
# Read-through/write-back during dataset generation: solved positions
# short-circuit repeated solves; fresh solves are persisted.
cargo run --release --bin cross_engine_benchmark -- dataset \
  --seed 20260711 --solve-budget 30.0 \
  --book benchmarks/results/book.db \
  --output benchmarks/positions-v1.json

# Bulk-export the canonical-representative solved references from an
# existing (checksum-verified) dataset artifact into a book. Idempotent:
# reruns upsert the same rows.
cargo run --release --bin cross_engine_benchmark -- export-book \
  --input benchmarks/positions-v1.json \
  --db benchmarks/results/book.db
```

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
